"""
오디오 전처리 파이프라인 v2 (yusung) — tmux 실행용 스크립트

`preprocess_data.ipynb`(v2로 갱신됨)와 완전히 동일한 로직을 .py로 옮긴 스크립트입니다.
노트북과의 유일한 차이는 "10초 미만 자투리 처리" 정책입니다 (자세한 내용은 preprocess_data.ipynb
상단 및 chunk_audio() 참고):

  - ASVspoof2019 LA / Common Voice KO / FakeMusicCaps : "single_chunk"
      파일당 정확히 1개의 10초 청크만 생성. 10초보다 길면 앞 10초만 사용(뒤는 버림),
      10초보다 짧으면 뒤를 0으로 패딩. (ASVspoof/CommonVoice는 원본 자체가 짧은 발화라 이게
      실제 데이터 특성이고, loop padding을 쓰면 같은 발화를 반복하는 비현실적 패턴이 생기므로 사용 안 함.
      FakeMusicCaps는 생성 자체가 약 10.0~10.24초라 뒤 자투리가 사실상 인코딩 tail이라 버림.)
  - MUSAN speech / MUSAN music : "sequential_drop"
      10초 단위로 순차 분할하고, 마지막 10초 미만 자투리는 패딩하지 않고 버림.
      원본이 충분히 길어(58~702초) 자투리를 버려도 데이터 손실이 미미함.
  - MUSAN noise : "noise_conditional"
      원본 길이가 10초 이상이면 sequential_drop과 동일하게, 10초 미만이면 single_chunk와
      동일하게(zero-pad 1개) 처리. noise는 길이 편차가 커서(0.7~398초) 파일별 조건부 처리가 필요.

기존 `yusung/data_processed/`(구버전: 자투리 무조건 zero-pad)는 건드리지 않고, 결과는 전부
`yusung/data_processed_v2/`에 새로 생성합니다.

실행 방법 (tmux 세션 안에서):
    conda activate deepvoice
    cd /cluster/workspace/deepvoice/yusung
    python3 preprocess_data_v2.py
"""

import importlib
import json
import math
import os
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for pkg in ["pyloudnorm"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import librosa
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from tqdm.auto import tqdm

# pyloudnorm은 게인 적용 후 peak가 1.0을 넘으면 미리 경고를 띄우는데, 우리는 바로 다음 단계
# (normalize_loudness의 peak_ceiling 클램프)에서 항상 그 초과분을 눌러주므로 안전하게 무시한다.
warnings.filterwarnings("ignore", message="Possible clipped samples in output")


# ============================================================
# 1. 설정값 (Config)
# ============================================================

# ===== 경로 =====
RAW_ROOT = Path("/cluster/workspace/deepvoice/data/raw")
OUT_ROOT = Path("/cluster/workspace/deepvoice/yusung/data_processed_v2")  # v1과 섞이지 않도록 별도 폴더
AUDIO_OUT = OUT_ROOT / "audio"        # 표준화된 오디오 (16k/mono/loudness-normalized/10s chunk)
META_OUT = OUT_ROOT / "metadata"      # 라벨 + split이 담긴 csv들
AUDIO_OUT.mkdir(parents=True, exist_ok=True)
META_OUT.mkdir(parents=True, exist_ok=True)

# ===== 오디오 표준화 =====
TARGET_SR = 16000                     # 평가 데이터 스펙: 16kHz
CHUNK_SEC = 10                        # 10초 단위 청크
CHUNK_SAMPLES = TARGET_SR * CHUNK_SEC

# ===== 라우드니스 정규화 (EBU R128 기반 LUFS) =====
TARGET_LUFS = -23.0                   # 방송 표준(EBU R128) 기본값. 모든 소스를 이 라우드니스로 통일.
PEAK_CEILING_DB = -1.0                # 정규화 후 true-peak이 이 값을 넘지 않도록 안전 마진(클리핑 방지)
PEAK_CEILING = 10 ** (PEAK_CEILING_DB / 20)

# ===== 10초 미만/이상 자투리 처리 정책 (v2에서 새로 추가) =====
# "single_chunk": 파일당 1개 청크만 생성, 길면 앞만 crop, 짧으면 zero-pad
# "sequential_drop": 10초 단위 순차 분할, 마지막 자투리는 버림(패딩 없음)
# "noise_conditional": 10초 이상이면 sequential_drop, 미만이면 single_chunk
CHUNK_POLICY_BY_KEY = {
    "asvspoof2019_la": "single_chunk",
    "commonvoice_ko": "single_chunk",
    "musan/speech": "sequential_drop",
    "musan/music": "sequential_drop",
    "musan/noise": "noise_conditional",
    "fakemusiccaps": "single_chunk",
}

# ===== train/val/test =====
SPLIT_RATIOS = (0.8, 0.1, 0.1)        # (train, val, test)
SPLIT_SEED = 42

# ===== 실행 =====
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)   # 공유 서버이므로 1코어는 남겨둠
MAX_FILES_PER_DATASET = None          # 디버그용 상한. 빠른 검증 시 예: 20. 실제 실행 시 반드시 None.

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

# ===== 메타데이터 컬럼 순서 + 체크포인트 파일 =====
COLUMN_ORDER = [
    "chunk_id", "filepath", "dataset", "split", "orig_split", "category", "source_model",
    "VP", "MP", "VF", "MF", "FF",
    "chunk_policy",
    "chunk_index", "n_chunks_in_file", "is_padded", "pad_seconds", "is_truncated",
    "orig_duration_sec", "orig_sr", "orig_channels",
    "measured_lufs", "target_lufs", "gain_applied_db", "peak_after_norm", "loudness_norm_applied",
    "group_id", "source_path",
]

# 처리된 chunk 메타데이터를 파일 단위로 즉시 append하는 체크포인트. 커널이 도중에 죽어도
# 여기까지 기록된 내용은 디스크에 남아있어서, 재실행 시 이미 끝난 파일을 다시 처리하지 않는다.
PARTIAL_CSV = META_OUT / "chunks_partial.csv"


# ============================================================
# 2. 오디오 처리 핵심 함수
# ============================================================

def probe_audio_info(path):
    """원본 파일의 샘플레이트/채널수/길이를 (오디오를 통째로 디코딩하지 않고) 빠르게 조회."""
    try:
        info = sf.info(str(path))
        duration = info.frames / info.samplerate if info.samplerate else None
        return {"samplerate": info.samplerate, "channels": info.channels, "duration_sec": duration}
    except Exception:
        return {"samplerate": None, "channels": None, "duration_sec": None}


def load_mono_16k(path, target_sr=TARGET_SR):
    """
    오디오를 mono/16kHz로 로드.
    librosa.load(sr=..., mono=True)가 리샘플링과 (스테레오 -> 채널 평균) 모노 변환을 함께 수행.
    soundfile 백엔드가 mp3/flac/wav/ogg/m4a를 모두 지원하므로 확장자별 분기가 필요 없다.
    """
    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return y.astype(np.float32), sr
    except Exception as e:
        return None, str(e)


def normalize_loudness(y, sr, target_lufs=TARGET_LUFS, peak_ceiling=PEAK_CEILING):
    """
    ITU-R BS.1770 (EBU R128) integrated loudness를 측정해 target_lufs로 맞춘다.
    이후 true peak이 peak_ceiling을 넘으면 선형으로 눌러서 클리핑을 방지한다
    (라우드니스가 target보다 살짝 낮아질 수 있지만, 이게 표준적인 "loudness normalize + peak limit" 방식).

    아주 짧거나(< 0.4초, BS.1770 게이팅 블록 최소 길이) 무음에 가까운 신호는 측정치가
    -inf/NaN이 되거나 예외가 나므로, 이런 경우 정규화를 건너뛰고 원신호를 그대로 반환한다
    (버리지 않고 살려서 chunk 단계로 넘긴다).
    """
    if y is None or len(y) < int(sr * 0.4):
        peak = float(np.max(np.abs(y))) if y is not None and len(y) else 0.0
        return y, None, 0.0, peak, False

    meter = pyln.Meter(sr)
    try:
        measured = meter.integrated_loudness(y)
    except Exception:
        return y, None, 0.0, float(np.max(np.abs(y))), False

    if not math.isfinite(measured):
        return y, None, 0.0, float(np.max(np.abs(y))), False

    gain_db = target_lufs - measured
    y_norm = pyln.normalize.loudness(y, measured, target_lufs)

    peak = float(np.max(np.abs(y_norm))) if len(y_norm) else 0.0
    if peak > peak_ceiling:
        y_norm = y_norm * (peak_ceiling / peak)
        peak = peak_ceiling

    return y_norm.astype(np.float32), float(measured), float(gain_db), peak, True


def chunk_audio(y, policy, chunk_samples=CHUNK_SAMPLES):
    """
    청크 분할 정책 3종 (v2에서 데이터셋별로 분기; v1은 전부 "항상 zero-pad, 절대 안 버림"이었음):

      - "single_chunk": 파일당 정확히 1개 청크만 생성. 10초보다 길면 앞 10초만 사용(뒤는 버림,
        is_truncated=True), 10초보다 짧으면 뒤를 0으로 패딩(is_padded=True). ASVspoof/CommonVoice
        (원본 자체가 짧은 실제 발화)와 FakeMusicCaps(생성 자체가 ~10.0~10.24초라 자투리가 사실상
        인코딩 tail)에 사용. 원본을 loop해서 반복 패턴을 만드는 대신, 짧으면 있는 그대로 zero-pad.
      - "sequential_drop": 10초 단위로 순차 분할하고, 마지막 10초 미만 자투리는 패딩하지 않고
        그냥 버림(청크를 만들지 않음). 원본이 충분히 길어(MUSAN speech/music, 58~702초) 자투리를
        버려도 데이터 손실이 미미할 때 사용.
      - "noise_conditional": 원본 길이가 10초 이상이면 sequential_drop과 동일하게, 10초 미만이면
        single_chunk와 동일하게 처리. MUSAN noise처럼 길이 편차가 커서(0.7~398초) 파일별 조건부
        처리가 필요한 경우 사용.

    반환: [(arr, is_padded, pad_seconds, is_truncated), ...]
    """

    def _single(y_):
        n = len(y_)
        if n == chunk_samples:
            return [(y_, False, 0.0, False)]
        if n > chunk_samples:
            return [(y_[:chunk_samples], False, 0.0, True)]
        pad_width = chunk_samples - n
        padded = np.pad(y_, (0, pad_width), mode="constant")
        return [(padded, True, pad_width / TARGET_SR, False)]

    def _sequential_drop(y_):
        n_full = len(y_) // chunk_samples
        if n_full == 0:
            # 안전장치: 이 정책 대상 데이터셋에서는 발생하지 않아야 하지만, 혹시 예상보다 짧은
            # 파일이 섞여 있어도 전체를 버리지 않고 zero-pad 1개로 보존한다.
            return _single(y_)
        chunks = [
            (y_[i * chunk_samples:(i + 1) * chunk_samples], False, 0.0, False)
            for i in range(n_full)
        ]
        # 마지막 자투리(remainder < chunk_samples)는 의도적으로 버림: 패딩도, 별도 청크 생성도 안 함.
        return chunks

    if policy == "single_chunk":
        return _single(y)
    if policy == "sequential_drop":
        return _sequential_drop(y)
    if policy == "noise_conditional":
        return _sequential_drop(y) if len(y) >= chunk_samples else _single(y)
    raise ValueError(f"unknown chunk_policy: {policy!r}")


def _process_file_job(job):
    """
    job = {"in_path", "out_dir", "base_name", "label": {..., "chunk_policy": ...}} 하나를 처리해서
    {"rows": [chunk별 메타데이터 dict, ...], "error": str | None} 을 반환.
    ProcessPoolExecutor.map에 그대로 넘길 수 있도록 순수 함수로 작성.
    """
    in_path = job["in_path"]
    out_dir = Path(job["out_dir"])
    base_name = job["base_name"]
    label = job["label"]
    chunk_policy = label["chunk_policy"]

    try:
        info = probe_audio_info(in_path)
        y, sr = load_mono_16k(in_path)
        if y is None or len(y) == 0:
            return {"rows": [], "error": f"{in_path}: load failed ({sr})"}

        y_norm, measured_lufs, gain_db, peak, norm_ok = normalize_loudness(y, sr)
        chunks = chunk_audio(y_norm, chunk_policy)

        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx, (arr, is_padded, pad_seconds, is_truncated) in enumerate(chunks):
            out_path = out_dir / f"{base_name}_chunk{idx:04d}.wav"
            arr = np.clip(arr, -1.0, 1.0)  # PCM_16 저장 전 최종 안전장치 (부동소수점 오차로 인한 오버슈트 방지)
            sf.write(str(out_path), arr, TARGET_SR, subtype="PCM_16")

            rel_path = out_path.relative_to(AUDIO_OUT)
            # chunk_id를 dataset+base_name만으로 만들면 FakeMusicCaps처럼 동일 파일명(원곡 id)이
            # 여러 하위 폴더(생성모델)에 반복되는 경우 충돌한다. 실제로 유일함이 보장된 상대경로에서
            # 그대로 파생시켜 folder 구분이 항상 반영되도록 한다.
            chunk_id = str(rel_path.with_suffix("")).replace("/", "__")

            row = dict(label)
            row.update(
                chunk_id=chunk_id,
                filepath=str(rel_path),  # AUDIO_OUT 기준 상대경로로 저장 (이식성)
                chunk_index=idx,
                n_chunks_in_file=len(chunks),
                is_padded=is_padded,
                pad_seconds=round(pad_seconds, 4),
                is_truncated=is_truncated,
                orig_duration_sec=info["duration_sec"],
                orig_sr=info["samplerate"],
                orig_channels=info["channels"],
                measured_lufs=measured_lufs,
                target_lufs=TARGET_LUFS,
                gain_applied_db=gain_db,
                peak_after_norm=peak,
                loudness_norm_applied=norm_ok,
            )
            rows.append(row)
        return {"rows": rows, "error": None}
    except Exception as e:
        return {"rows": [], "error": f"{in_path}: {e}"}


def run_jobs(jobs, desc, num_workers=NUM_WORKERS):
    """job 리스트를 (병렬로) 처리하고 모든 chunk 행을 하나의 리스트로 모아 반환."""
    all_rows, errors = [], []
    if num_workers <= 1:
        results = (_process_file_job(j) for j in jobs)
        for res in tqdm(results, total=len(jobs), desc=desc):
            all_rows.extend(res["rows"])
            if res["error"]:
                errors.append(res["error"])
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for res in tqdm(ex.map(_process_file_job, jobs, chunksize=8), total=len(jobs), desc=desc):
                all_rows.extend(res["rows"])
                if res["error"]:
                    errors.append(res["error"])
    if errors:
        print(f"[{desc}] {len(errors)}개 실패, 예시:", errors[:3])
    return all_rows


def load_done_source_paths():
    """체크포인트(PARTIAL_CSV)에 이미 기록된 source_path 집합을 반환. 재실행 시 이 파일들은 건너뛴다."""
    if not PARTIAL_CSV.exists():
        return set()
    return set(pd.read_csv(PARTIAL_CSV, usecols=["source_path"])["source_path"])


def append_rows_checkpoint(rows):
    """새로 처리된 chunk 행들을 PARTIAL_CSV에 즉시 append (파일이 없을 때만 헤더를 쓴다).
    이 함수가 끝나는 순간 디스크에 기록되므로, 그 이후 프로세스가 죽어도 이 rows는 재계산할 필요가 없다."""
    if not rows:
        return
    chunk_df = pd.DataFrame(rows)[COLUMN_ORDER]
    chunk_df.to_csv(PARTIAL_CSV, mode="a", header=not PARTIAL_CSV.exists(), index=False)


# ============================================================
# 3. ASVspoof 2019 LA
# ============================================================

def build_and_run_asvspoof():
    ASV_ROOT = RAW_ROOT / "asvspoof2019_la" / "extracted" / "LA"
    ASV_PROTOCOLS = ASV_ROOT / "ASVspoof2019_LA_cm_protocols"

    ASV_SPLITS = {
        "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt", "train"),
        "dev":   ("ASVspoof2019_LA_dev",   "ASVspoof2019.LA.cm.dev.trl.txt",   "val"),
        "eval":  ("ASVspoof2019_LA_eval",  "ASVspoof2019.LA.cm.eval.trl.txt",  "test"),
    }

    done = load_done_source_paths()

    asv_rows = []
    asv_unmapped = 0
    asv_skipped = 0

    for orig_split, (folder, proto_name, mapped_split) in ASV_SPLITS.items():
        proto_df = pd.read_csv(
            ASV_PROTOCOLS / proto_name, sep=r"\s+", header=None,
            names=["speaker_id", "file_id", "unused", "system_id", "key"],
        )
        proto_map = {r.file_id: (r.key, r.speaker_id) for r in proto_df.itertuples()}

        flac_dir = ASV_ROOT / folder / "flac"
        files = sorted(flac_dir.glob("*.flac"))[:MAX_FILES_PER_DATASET]

        jobs = []
        for f in files:
            if str(f) in done:
                asv_skipped += 1
                continue
            info = proto_map.get(f.stem)
            if info is None:
                asv_unmapped += 1
                continue
            key, speaker_id = info
            is_spoof = int(key == "spoof")
            label = dict(
                dataset="asvspoof2019_la", VP=1, MP=0, VF=is_spoof, MF=-1, FF=is_spoof,
                source_model=None, category=key,
                chunk_policy=CHUNK_POLICY_BY_KEY["asvspoof2019_la"],
                orig_split=orig_split, split=mapped_split,
                group_id=f"asvspoof2019_la::{speaker_id}", source_path=str(f),
            )
            jobs.append(dict(
                in_path=str(f), out_dir=str(AUDIO_OUT / "asvspoof2019_la" / orig_split),
                base_name=f.stem, label=label,
            ))

        new_rows = run_jobs(jobs, desc=f"asvspoof2019_la/{orig_split}")
        append_rows_checkpoint(new_rows)
        asv_rows += new_rows

    print("asvspoof2019_la 신규 chunk 수:", len(asv_rows), "| 이어서 건너뜀:", asv_skipped, "| protocol 매칭 실패:", asv_unmapped)
    return asv_rows


# ============================================================
# 4. Common Voice Korean
# ============================================================

def build_and_run_commonvoice():
    CV_ROOT = RAW_ROOT / "commonvoice_ko" / "cv-corpus-26.0-2026-06-12" / "ko"

    tsv_files = ["validated.tsv", "train.tsv", "dev.tsv", "test.tsv", "other.tsv", "invalidated.tsv"]
    lookup_frames = [
        pd.read_csv(CV_ROOT / name, sep="\t", usecols=["client_id", "path"])
        for name in tsv_files if (CV_ROOT / name).exists()
    ]
    client_lookup = (
        pd.concat(lookup_frames)
        .drop_duplicates(subset="path", keep="first")
        .set_index("path")["client_id"]
        .to_dict()
    )

    official_split_map = {}
    for name, mapped in [("train.tsv", "train"), ("dev.tsv", "val"), ("test.tsv", "test")]:
        p = CV_ROOT / name
        if p.exists():
            for path in pd.read_csv(p, sep="\t", usecols=["path"])["path"]:
                official_split_map[path] = mapped

    done = load_done_source_paths()

    cv_files = sorted((CV_ROOT / "clips").glob("*.mp3"))[:MAX_FILES_PER_DATASET]
    cv_jobs = []
    cv_skipped = 0
    for f in cv_files:
        if str(f) in done:
            cv_skipped += 1
            continue
        client_id = client_lookup.get(f.name, f.stem)
        split = official_split_map.get(f.name)
        label = dict(
            dataset="commonvoice_ko", VP=1, MP=0, VF=0, MF=-1, FF=0,
            source_model=None, category=None,
            chunk_policy=CHUNK_POLICY_BY_KEY["commonvoice_ko"],
            orig_split=("cv_official" if split else None), split=split,
            group_id=f"commonvoice_ko::{client_id}", source_path=str(f),
        )
        cv_jobs.append(dict(
            in_path=str(f), out_dir=str(AUDIO_OUT / "commonvoice_ko"), base_name=f.stem, label=label,
        ))

    cv_rows = run_jobs(cv_jobs, desc="commonvoice_ko")
    append_rows_checkpoint(cv_rows)
    print("commonvoice_ko 신규 chunk 수:", len(cv_rows), "| 이어서 건너뜀:", cv_skipped)
    return cv_rows


# ============================================================
# 5. MUSAN
# ============================================================

def build_and_run_musan():
    MUSAN_ROOT = RAW_ROOT / "musan" / "extracted" / "musan"
    MUSAN_RULES = {
        "speech": dict(VP=1, MP=0, VF=0, MF=-1, FF=0, chunk_policy=CHUNK_POLICY_BY_KEY["musan/speech"]),
        "music":  dict(VP=0, MP=1, VF=-1, MF=0, FF=0, chunk_policy=CHUNK_POLICY_BY_KEY["musan/music"]),
        "noise":  dict(VP=0, MP=0, VF=-1, MF=-1, FF=0, chunk_policy=CHUNK_POLICY_BY_KEY["musan/noise"]),
    }

    done = load_done_source_paths()

    musan_rows = []
    musan_skipped = 0
    for category, rule in MUSAN_RULES.items():
        files = sorted((MUSAN_ROOT / category).rglob("*.wav"))[:MAX_FILES_PER_DATASET]
        jobs = []
        for f in files:
            if str(f) in done:
                musan_skipped += 1
                continue
            subsource = f.parent.name  # 예: librivox, us-gov, fma, free-sound ...
            label = dict(
                dataset="musan", **rule,
                source_model=None, category=f"{category}/{subsource}",
                orig_split=None, split=None,
                group_id=f"musan::{f.stem}", source_path=str(f),
            )
            jobs.append(dict(
                in_path=str(f), out_dir=str(AUDIO_OUT / "musan" / category / subsource),
                base_name=f.stem, label=label,
            ))
        new_rows = run_jobs(jobs, desc=f"musan/{category}")
        append_rows_checkpoint(new_rows)
        musan_rows += new_rows

    print("musan 신규 chunk 수:", len(musan_rows), "| 이어서 건너뜀:", musan_skipped)
    return musan_rows


# ============================================================
# 6. FakeMusicCaps
# ============================================================

def build_and_run_fakemusiccaps():
    FMC_ROOT = RAW_ROOT / "fakemusiccaps" / "processed"
    done = load_done_source_paths()

    fmc_jobs = []
    fmc_skipped = 0
    for model_dir in sorted(d for d in FMC_ROOT.iterdir() if d.is_dir() and d.name != "__MACOSX"):
        files = sorted(model_dir.glob("*.wav"))[:MAX_FILES_PER_DATASET]
        for f in files:
            if str(f) in done:
                fmc_skipped += 1
                continue
            label = dict(
                dataset="fakemusiccaps", VP=0, MP=1, VF=-1, MF=1, FF=1,
                source_model=model_dir.name, category=None,
                chunk_policy=CHUNK_POLICY_BY_KEY["fakemusiccaps"],
                orig_split=None, split=None,
                group_id=f"fakemusiccaps::{f.stem}", source_path=str(f),
            )
            fmc_jobs.append(dict(
                in_path=str(f), out_dir=str(AUDIO_OUT / "fakemusiccaps" / model_dir.name),
                base_name=f.stem, label=label,
            ))

    fmc_rows = run_jobs(fmc_jobs, desc="fakemusiccaps")
    append_rows_checkpoint(fmc_rows)
    print("fakemusiccaps 신규 chunk 수:", len(fmc_rows), "| 이어서 건너뜀:", fmc_skipped)
    return fmc_rows


# ============================================================
# 7. 통합 + train/val/test 최종 배정
# ============================================================

def grouped_split(groups: pd.Series, ratios=SPLIT_RATIOS, seed=SPLIT_SEED):
    """group_id 기준으로 (같은 그룹은 항상 같은 split에 가도록) train/val/test를 배정."""
    sizes = groups.value_counts()
    order = sizes.index.to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(order)

    total = sizes.sum()
    names = ["train", "val", "test"]
    targets = {n: r * total for n, r in zip(names, ratios)}
    assigned = {n: 0 for n in names}

    mapping = {}
    for g in order:
        need = {n: (assigned[n] / targets[n] if targets[n] > 0 else 1.0) for n in names}
        choice = min(names, key=lambda n: need[n])
        mapping[g] = choice
        assigned[choice] += sizes[g]
    return mapping


def merge_and_split():
    df = pd.read_csv(PARTIAL_CSV)
    print("전체 chunk 수 (누적):", df.shape)
    print("split 미배정 행 수:", df["split"].isna().sum())

    for dataset_name, seed_offset in [("commonvoice_ko", 1), ("musan", 2), ("fakemusiccaps", 3)]:
        mask = (df["dataset"] == dataset_name) & (df["split"].isna())
        if mask.any():
            mapping = grouped_split(df.loc[mask, "group_id"], seed=SPLIT_SEED + seed_offset)
            df.loc[mask, "split"] = df.loc[mask, "group_id"].map(mapping)

    assert df["split"].isna().sum() == 0, "split이 배정되지 않은 행이 남아있음"
    df = df[COLUMN_ORDER]

    print(pd.crosstab(df["dataset"], df["split"]))
    return df


# ============================================================
# 8. 검증(sanity check)
# ============================================================

def sanity_check(df):
    assert set(df["VP"].unique()) <= {0, 1}
    assert set(df["MP"].unique()) <= {0, 1}
    assert set(df["VF"].unique()) <= {-1, 0, 1}
    assert set(df["MF"].unique()) <= {-1, 0, 1}
    assert set(df["FF"].unique()) <= {0, 1}
    assert set(df["split"].unique()) <= {"train", "val", "test"}
    assert set(df["chunk_policy"].unique()) <= {"single_chunk", "sequential_drop", "noise_conditional"}
    assert set(df["is_truncated"].unique()) <= {True, False}

    missing = df[~df["filepath"].apply(lambda p: (AUDIO_OUT / p).exists())]
    assert missing.empty, f"{len(missing)}개 파일이 메타데이터에는 있지만 디스크에 없음"

    # 모든 청크는 정책과 무관하게 항상 정확히 CHUNK_SAMPLES 길이여야 한다
    # (single_chunk: crop-or-pad, sequential_drop: 완전한 청크만 생성).
    sample_paths = df["filepath"].sample(min(200, len(df)), random_state=0)
    for p in sample_paths:
        y, sr = sf.read(str(AUDIO_OUT / p))
        assert sr == TARGET_SR, (p, sr)
        assert len(y) == CHUNK_SAMPLES, (p, len(y))
    print(f"{len(sample_paths)}개 청크 길이/샘플레이트 검증 통과")

    print(df.groupby(["dataset", "split", "VP", "MP", "VF", "MF", "FF"], dropna=False).size())


# ============================================================
# 9. 메타데이터 저장
# ============================================================

def save_outputs(df):
    CONFIG = dict(
        target_sr=TARGET_SR,
        chunk_sec=CHUNK_SEC,
        target_lufs=TARGET_LUFS,
        peak_ceiling_db=PEAK_CEILING_DB,
        split_ratios=dict(zip(["train", "val", "test"], SPLIT_RATIOS)),
        split_seed=SPLIT_SEED,
        max_files_per_dataset=MAX_FILES_PER_DATASET,
        chunk_policy_by_dataset=CHUNK_POLICY_BY_KEY,
        chunk_policy_definitions={
            "single_chunk": (
                "파일당 정확히 1개의 10초 청크만 생성. 10초보다 길면 앞 10초만 사용(뒤는 버림, "
                "is_truncated=True), 10초보다 짧으면 뒤를 0으로 패딩(is_padded=True). 짧은 원본을 "
                "loop해서 반복 패턴을 인위적으로 만들지 않기 위함."
            ),
            "sequential_drop": (
                "10초 단위로 순차 분할하고, 마지막 10초 미만 자투리는 패딩하지 않고 버림 "
                "(청크를 만들지 않음). 원본이 충분히 길어 자투리를 버려도 손실이 미미할 때 사용."
            ),
            "noise_conditional": (
                "원본 길이가 10초 이상이면 sequential_drop과 동일하게, 10초 미만이면 single_chunk와 "
                "동일하게(zero-pad) 처리. 길이 편차가 큰 데이터셋(MUSAN noise)에 사용."
            ),
        },
        n_chunks_total=len(df),
        n_source_files=int(df.groupby("dataset")["source_path"].nunique().sum()),
    )

    df.to_csv(META_OUT / "chunks.csv", index=False)
    for split_name in ["train", "val", "test"]:
        df[df["split"] == split_name].to_csv(META_OUT / f"chunks_{split_name}.csv", index=False)

    df[df["VP"] == 1].to_csv(META_OUT / "voice_pool.csv", index=False)
    df[df["MP"] == 1].to_csv(META_OUT / "music_pool.csv", index=False)

    with open(META_OUT / "preprocessing_config.json", "w") as f:
        json.dump(CONFIG, f, indent=2, ensure_ascii=False)

    print("저장 완료:", META_OUT)
    print(json.dumps(CONFIG, indent=2, ensure_ascii=False))


# ============================================================
# main
# ============================================================

def main():
    print(f"TARGET_SR={TARGET_SR}, CHUNK_SEC={CHUNK_SEC}, TARGET_LUFS={TARGET_LUFS}, "
          f"NUM_WORKERS={NUM_WORKERS}, MAX_FILES_PER_DATASET={MAX_FILES_PER_DATASET}")
    print("OUT_ROOT:", OUT_ROOT)
    if PARTIAL_CSV.exists():
        n_done = sum(1 for _ in open(PARTIAL_CSV)) - 1
        print(f"체크포인트 발견: {PARTIAL_CSV} ({n_done}개 chunk 기록됨) -> 해당 파일들은 자동으로 건너뜁니다.")

    build_and_run_asvspoof()
    build_and_run_commonvoice()
    build_and_run_musan()
    build_and_run_fakemusiccaps()

    df = merge_and_split()
    sanity_check(df)
    save_outputs(df)


if __name__ == "__main__":
    main()
