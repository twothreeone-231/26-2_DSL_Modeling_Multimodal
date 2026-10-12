"""
[목적]
딥보이스 탐지 웹앱의 백엔드 서버(FastAPI).
- 프론트엔드 정적 파일(HTML/CSS/JS) 제공
- POST /analyze : 업로드된 오디오를 분석해 JSON 결과 반환
- GET  /health  : 서버가 살아있는지 + 현재 분석기 모드 확인

[실행]
    cd backend
    uvicorn main:app --host 0.0.0.0 --port 18342

[설계 메모 1 — 프론트/백을 한 서버에서 제공하는 이유]
브라우저는 보안상 "다른 출처(주소·포트)로의 요청"을 기본 차단한다(CORS).
프론트를 별도 포트로 띄우면 CORS 설정이 필요한데, 이는 입문 단계의 대표적인
삽질 구간이다. 같은 서버가 HTML도 주고 API도 주면 출처가 같아져서 이 문제가
아예 발생하지 않는다. 덤으로 SSH 터널도 포트 하나만 뚫으면 된다.

[설계 메모 2 — 업로드 파일을 저장하지 않는 이유]
음성은 개인정보라서 서버에 남기는 순간 처리방침·동의 절차가 필요해진다.
임시 파일로 받아 분석 직후 삭제하면 이 문제가 대부분 사라지고, 사용자에게도
"저장하지 않는다"고 명시할 수 있어 신뢰가 올라간다.

[설계 메모 3 — --host 0.0.0.0 이 필요한 이유]
기본값(127.0.0.1)은 "그 컴퓨터 안에서만" 접속을 받는다. SLURM 계산 노드에서
띄우고 노트북에서 SSH 터널로 접속하려면 외부 인터페이스에도 열어야 하므로
0.0.0.0이 필요하다. 맥북 로컬 실행 때는 127.0.0.1이어도 무방하다.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from analyzer import DummyAnalyzer

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"

# 파이프라인이 지원하는 확장자 (src/audio_io.py의 목록과 동일하게 유지할 것)
ALLOWED_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma", ".webm"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50MB. 통화 녹음 기준 충분하고, 서버 메모리도 보호한다.

# [분석기 선택] 환경변수로 바꿀 수 있게 해둔 이유: 모델을 붙인 뒤에도
# 프론트엔드만 고칠 때는 무거운 모델 로딩(약 27초) 없이 dummy로 빠르게 띄우고 싶기 때문.
ANALYZER_MODE = os.environ.get("ANALYZER_MODE", "dummy")

if ANALYZER_MODE == "dummy":
    analyzer = DummyAnalyzer()

elif ANALYZER_MODE == "model":
    from analyzer import PipelineAnalyzer

    # [경로를 환경변수로 받는 이유] 계산 노드에서 코드와 가중치가 서로 다른
    # 디렉터리에 흩어져 있다(코드는 jaehyun/, DF-Arena는 model/, HF 캐시는
    # checkpoints/). src/config.py가 이미 JAEHYUN_* 환경변수로 경로를 받도록
    # 설계돼 있어, 코드를 고치지 않고 연결할 수 있다. run_on_slurm.sh 참고.
    PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT")
    if not PIPELINE_ROOT:
        raise RuntimeError(
            "PIPELINE_ROOT 환경변수가 필요합니다 (src/pipeline.py가 있는 폴더의 절대 경로). "
            "run_on_slurm.sh를 사용하세요."
        )
    if not (Path(PIPELINE_ROOT) / "src" / "pipeline.py").is_file():
        raise RuntimeError(f"PIPELINE_ROOT에 src/pipeline.py가 없습니다: {PIPELINE_ROOT}")

    analyzer = PipelineAnalyzer(
        pipeline_root=PIPELINE_ROOT,
        device_str=os.environ.get("PIPELINE_DEVICE", "cuda"),
    )

else:
    raise RuntimeError(f"알 수 없는 ANALYZER_MODE입니다: {ANALYZER_MODE} (dummy 또는 model)")

app = FastAPI(title="딥보이스 탐지 데모", docs_url="/docs")


@app.get("/health")
def health():
    """[목적] 서버 상태 확인용. 터널이 제대로 뚫렸는지 브라우저로 바로 볼 수 있다."""
    return {"ok": True, "analyzer_mode": analyzer.mode}


@app.post("/analyze")
async def analyze(audio: UploadFile = File(...)):
    """
    [목적] 오디오 파일을 받아 분석 결과 JSON을 반환한다.

    [multipart/form-data로 받는 이유] JSON은 텍스트 형식이라 바이너리(오디오)를
    그대로 담을 수 없다. 파일 업로드는 multipart 형식이 표준이며, 브라우저의
    FormData 객체가 이 형식을 자동으로 만들어준다.
    """
    suffix = Path(audio.filename or "").suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 형식입니다: {suffix}")

    # 임시 파일로 받는다. 분석기가 파일 경로를 받는 인터페이스이기도 하고,
    # 큰 파일을 통째로 메모리에 들고 있지 않기 위해서이기도 하다.
    tmp_dir = Path(tempfile.mkdtemp(prefix="deepvoice_"))
    tmp_path = tmp_dir / f"upload{suffix or '.bin'}"
    try:
        size = 0
        with tmp_path.open("wb") as f:
            while chunk := await audio.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="파일이 너무 큽니다 (최대 50MB)")
                f.write(chunk)

        if size == 0:
            raise HTTPException(status_code=400, detail="빈 파일입니다")

        result = analyzer.analyze(tmp_path, original_name=audio.filename or "")
        result["filename"] = audio.filename
        result["size_bytes"] = size
        return JSONResponse(result)

    except HTTPException:
        raise
    except Exception as e:
        # 분석 실패가 서버를 죽이지 않게 한다. 어떤 오디오가 들어올지 모르기 때문.
        return JSONResponse(
            status_code=500,
            content={"ok": False, "error": f"{type(e).__name__}: {e}"},
        )
    finally:
        # 설계 메모 2 참고: 성공하든 실패하든 업로드본은 즉시 삭제한다.
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.get("/")
def index():
    """[목적] 루트 접속 시 프론트엔드 첫 화면을 내려준다."""
    return FileResponse(FRONTEND_DIR / "index.html")


# [마운트 순서 주의] StaticFiles를 "/"에 붙이면 위에서 정의한 라우트를 가려버린다.
# 그래서 정적 파일은 /static 아래로 따로 두고, HTML에서 /static/... 으로 참조한다.
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
