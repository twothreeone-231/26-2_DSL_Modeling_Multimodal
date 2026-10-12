"""
[목적]
오디오 1개를 받아 "AI 생성 여부 분석 결과"를 돌려주는 분석기 모듈.

[현재 단계] 실제 모델은 아직 연결하지 않았다. DummyAnalyzer가 그럴듯한 가짜
결과를 돌려주고, 프론트엔드 흐름을 모델 없이 완성하는 데 쓴다.

[설계 메모 — 왜 이렇게 분리했는가]
분석기를 main.py에서 떼어내고 `analyze(path) -> dict` 라는 인터페이스 하나로
고정했다. 나중에 SLURM 계산 노드에서 실제 파이프라인을 붙일 때
PipelineAnalyzer를 추가하고 main.py에서 어떤 구현을 쓸지만 바꾸면 되며,
프론트엔드는 한 줄도 수정할 필요가 없다.

[반환 JSON 형식을 실제 파이프라인에 맞춘 이유]
아래 키 구성은 대회 파이프라인 `AudioPresencePipeline.analyze_file()`이
내놓는 값들을 그대로 옮긴 것이다(voice/music presence, 각 성분 fake 확률,
라우팅 결과, 분리 여부). 지금부터 형식을 맞춰두지 않으면 모델을 붙이는
시점에 프론트엔드까지 연쇄 수정이 필요해진다.
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path


# 분석 결과에 항상 따라붙는 모델 한계 경고.
# [왜 서버에서 내려보내는가] 화면에만 박아두면 나중에 모델이 바뀌었을 때
# 프론트엔드와 실제 성능이 어긋날 수 있다. 성능 수치를 아는 쪽(백엔드)이
# 같이 내려보내는 편이 일관성이 유지된다.
MODEL_DISCLAIMER = (
    "이 모델은 참고용 보조 지표입니다. 음성 판정의 동일오류율(EER)이 약 22%로, "
    "다섯 건 중 한 건 정도는 틀릴 수 있습니다. 확률이 낮게 나와도 안전하다는 뜻이 아니며, "
    "아래 정황 체크 결과를 우선으로 판단하세요."
)


class DummyAnalyzer:
    """[목적] 모델 없이 프론트엔드를 개발/시연하기 위한 가짜 분석기."""

    mode = "dummy"

    # [난수 대신 파일명 해시를 쓰는 이유] 같은 파일을 넣으면 항상 같은 결과가
    # 나와야 시연 중에 화면이 예측 가능하고, 버그 재현도 쉽다.
    SCENARIOS = [
        # (음성존재, 음악존재, 음성fake, 음악fake, 분리실행, 라우팅라벨)
        (0.96, 0.04, 0.88, 0.05, False, "MUSIC_ONLY"),
        (0.93, 0.71, 0.74, 0.38, True, "BOTH"),
        (0.89, 0.12, 0.21, 0.09, True, "VOICE_ONLY_SEPARATED"),
        (0.07, 0.95, 0.11, 0.66, False, "MUSIC_ONLY"),
    ]

    def __init__(self, fake_latency_sec: float = 2.5):
        # 실제 모델은 GPU에서 파일당 ~4초가 걸린다. 지연이 0이면 로딩 UI와
        # "분석 대기 중 체크리스트 작성" 흐름을 테스트할 수 없어서 일부러 넣는다.
        self.fake_latency_sec = fake_latency_sec

    def analyze(self, audio_path: Path, original_name: str = "") -> dict:
        """[목적] 오디오 경로를 받아 분석 결과 dict를 반환한다."""
        t0 = time.time()
        time.sleep(self.fake_latency_sec)

        key = original_name or audio_path.name
        idx = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(self.SCENARIOS)
        voice_p, music_p, voice_f, music_f, separated, route = self.SCENARIOS[idx]

        # 실측(RTX 6000 Ada) 단계별 시간을 흉내낸 값. 분리 실행 여부에 따라 달라진다.
        stages = [
            ("오디오 로딩", 0.04),
            ("AudioMosaic 성분 검출", 0.21),
            ("BS PolarFormer 음원 분리", 2.90 if separated else None),
            ("AudioMosaic 분리본 재검출", 0.42 if separated else None),
            ("DF-Arena 음성 판정", 0.18),
            ("lofcz 음악 판정", 0.33),
        ]

        return build_result(
            mode=self.mode,
            voice_present=voice_p,
            music_present=music_p,
            voice_fake=voice_f,
            music_fake=music_f,
            separated=separated,
            route=route,
            elapsed=time.time() - t0,
            stages=stages,
            zones={
                "voice": "HIGH" if voice_p >= 0.6 else "MID" if voice_p >= 0.3 else "LOW",
                "music": "HIGH" if music_p >= 0.6 else "MID" if music_p >= 0.3 else "LOW",
            },
        )


class PipelineAnalyzer:
    """
    [목적] 실제 대회 파이프라인(AudioPresencePipeline)을 감싸 웹 응답 형식으로 변환한다.

    [모델을 서버 시작 시 1회만 로드하는 이유] 계산 노드 실측 기준 로딩에 약 27초가
    걸린다(DF-Arena 4.3GB 포함). 요청마다 로드하면 쓸 수 없고, 지연 로드를 하면
    첫 사용자가 27초를 기다린다. 발표 전에 미리 띄워 워밍업하는 운영 방식과도 맞다.

    [실측 성능] RTX 6000 Ada 기준 로딩 26.9초 / 분석 0.78초(분리 미실행) /
    GPU peak 약 5GB. 음악이 섞여 분리가 실행되면 3~4초대로 늘어난다.
    """

    mode = "model"

    def __init__(self, pipeline_root: str, device_str: str = "cuda"):
        import torch

        # 파이프라인 코드를 import 하려면 프로젝트 루트가 sys.path에 있어야 한다.
        if pipeline_root not in sys.path:
            sys.path.insert(0, pipeline_root)

        from src import config  # noqa: E402
        from src.pipeline import AudioPresencePipeline  # noqa: E402

        cfg = config.PipelineConfig()
        # 대회 제출과 동일한 구성으로 맞춘다. 발표에서 "제출한 모델 그대로"라고
        # 말할 수 있어야 하고, 성능 수치(EER)도 이 구성에서 측정된 값이기 때문.
        cfg.enable_artifactnet = True
        cfg.music_fake_backend = "lofcz"
        cfg.lofcz.calibration_scale = 0.142962
        cfg.lofcz.calibration_bias = 3.097933
        cfg.enable_voice_fake_detection = True

        t0 = time.time()
        print(f"[분석기] 파이프라인 로딩 시작 (약 30초 소요)... root={pipeline_root}", flush=True)
        self.pipeline = AudioPresencePipeline(cfg=cfg, device=torch.device(device_str))
        print(f"[분석기] 로딩 완료 ({time.time() - t0:.1f}초)", flush=True)

    def analyze(self, audio_path: Path, original_name: str = "") -> dict:
        """[목적] 오디오 1건을 실제 모델로 분석하고 웹 응답 형식으로 변환한다."""
        # [side conformer 리셋] jaehyun outputs/probe_voice_paired.zip의
        # script.py와 동일한 패턴 -- 이전 요청의 값이 남아있으면 이번 파일이
        # 분리(separated)되지 않았을 때 지난 값을 잘못 재사용하게 된다.
        voice_detector = self.pipeline.voice_fake_detector
        if voice_detector is not None:
            voice_detector.last_side_probability = None

        r = self.pipeline.analyze_file(audio_path)
        diag = r.get("diagnostics", {})
        timings = diag.get("timings", {})
        separated = bool(diag.get("separated", False))

        # [None 처리] 탐지기가 비활성화된 경우에만 None이 나온다. 지금 구성에서는
        # 둘 다 켜져 있어 발생하지 않지만, 방어적으로 0.0 + 신뢰불가로 처리한다.
        voice_fake = r.get("voice_fake_probability")
        music_fake = r.get("music_fake_probability")

        # [component-isolation probe] 분리(separated)가 실행된 파일에 한해,
        # 파인튜닝된 side Conformer 점수도 같이 보여준다 (공개 모델 점수는
        # 그대로 유지 -- DACON 제출 로직과 동일하게 참고용 비교치로만 노출).
        side_prob = (
            getattr(voice_detector, "last_side_probability", None) if separated else None
        )

        return build_result(
            mode=self.mode,
            voice_present=r["voice_probability"],
            music_present=r["music_probability"],
            voice_fake=voice_fake if voice_fake is not None else 0.0,
            music_fake=music_fake if music_fake is not None else 0.0,
            separated=bool(diag.get("separated", False)),
            route=r.get("route", "-"),
            elapsed=timings.get("total", 0.0),
            # 단계별 소요 시간은 데모에서 파이프라인 구조를 보여주는 데 쓴다.
            stages=[
                ("오디오 로딩", timings.get("audio_load")),
                ("AudioMosaic 성분 검출", timings.get("audiomosaic_original")),
                ("BS PolarFormer 음원 분리", timings.get("polarformer_separation")),
                ("AudioMosaic 분리본 재검출", timings.get("audiomosaic_stems")),
                ("DF-Arena 음성 판정", timings.get("voice_fake_detection")),
                ("lofcz 음악 판정", timings.get("artifactnet")),
            ],
            zones={"voice": diag.get("voice_zone"), "music": diag.get("music_zone")},
            voice_fake_none=voice_fake is None,
            music_fake_none=music_fake is None,
            voice_fake_side=side_prob,
        )


def build_result(
    *,
    mode: str,
    voice_present: float,
    music_present: float,
    voice_fake: float,
    music_fake: float,
    separated: bool,
    route: str,
    elapsed: float,
    stages: list | None = None,
    zones: dict | None = None,
    voice_fake_none: bool = False,
    music_fake_none: bool = False,
    voice_fake_side: float | None = None,
) -> dict:
    """
    [목적] 분석기 구현이 무엇이든 동일한 형태의 응답 JSON을 만들어주는 헬퍼.
    실제 모델을 붙일 때도 이 함수를 재사용하면 프론트엔드 호환이 자동으로 보장된다.
    """
    # [성분별 신뢰 가중] 존재 확률이 낮은 성분의 fake 점수는 의미가 없으므로
    # 화면에 "참고 불가"로 표시할 수 있도록 플래그를 같이 내려준다.
    # 기준값 0.3은 대회 파이프라인의 라우팅 임계값(RoutingConfig.low_threshold)과
    # 같은 값으로 맞췄다.
    PRESENCE_MIN = 0.3

    return {
        "ok": True,
        "mode": mode,  # "dummy" | "model" | "cached"
        "presence": {
            "voice": round(voice_present, 4),
            "music": round(music_present, 4),
        },
        "fake": {
            "voice": round(voice_fake, 4),
            "music": round(music_fake, 4),
            # [실험용 비교치] 음성이 분리(separated)된 파일에서만 채워진다.
            # 공개 DF-Arena 모델과 같은 입력을 파인튜닝된 side Conformer로도
            # 점수 매긴 값 -- 참고용이며, 실제 판정(voice)은 항상 공개 모델 값이다.
            "voice_side_probe": round(voice_fake_side, 4) if voice_fake_side is not None else None,
        },
        "reliable": {
            # 해당 성분이 충분히 존재해서 fake 점수를 믿을 만한지
            "voice": voice_present >= PRESENCE_MIN and not voice_fake_none,
            "music": music_present >= PRESENCE_MIN and not music_fake_none,
        },
        "pipeline": {
            "separated": separated,   # BS PolarFormer 분리를 돌렸는지
            "route": route,           # 라우팅 결정 라벨
            "zones": zones or {},     # voice/music 각각의 LOW/MID/HIGH 판정
            # 단계별 소요 시간 (값이 있는 단계만). 데모에서 파이프라인 구조를 드러내는 용도.
            "stages": [
                {"name": n, "sec": round(s, 3)}
                for n, s in (stages or [])
                if s is not None and s > 0
            ],
        },
        "elapsed_sec": round(elapsed, 2),
        "disclaimer": MODEL_DISCLAIMER,
    }
