#!/usr/bin/env python3
"""
v3 오디오 전처리 파이프라인
- 4초 미만 청크 필터링 추가
- train/val/test 폴더 구조로 직접 저장
- 메타데이터를 split별로 CSV로 기록

Usage:
  python preprocess_data_v3.py
  또는 tmux에서:
  tmux new-session -d -s preprocess_v3 'cd /cluster/workspace/deepvoice && python yusung/preprocess_data_v3.py'
"""

import importlib
import json
import math
import os
import re
import subprocess
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

# 필요한 패키지 확인 및 설치
for pkg in ["librosa", "numpy", "pandas", "pyloudnorm", "soundfile"]:
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"[INFO] {pkg} 설치 중...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=True)

import librosa
import numpy as np
import pandas as pd
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

warnings.filterwarnings("ignore", message="Possible clipped samples in output")

# ===== 경로 =====
RAW_ROOT = Path("/cluster/workspace/deepvoice/data/raw")
OUT_ROOT = Path("/cluster/workspace/deepvoice/yusung/data_processed_v3")
AUDIO_OUT = OUT_ROOT / "audio"
META_OUT = OUT_ROOT / "metadata"
AUDIO_OUT.mkdir(parents=True, exist_ok=True)
META_OUT.mkdir(parents=True, exist_ok=True)

# ===== 오디오 표준화 =====
TARGET_SR = 16000
CHUNK_SEC = 10
CHUNK_SAMPLES = TARGET_SR * CHUNK_SEC

# v3에서 추가: 4초 미만 필터
MIN_CHUNK_SEC = 4.0
MIN_CHUNK_SAMPLES = int(TARGET_SR * MIN_CHUNK_SEC)

# ===== 라우드니스 정규화 =====
TARGET_LUFS = -23.0
PEAK_CEILING_DB = -1.0
PEAK_CEILING = 10 ** (PEAK_CEILING_DB / 20)

# ===== 청크 정책 =====
CHUNK_POLICY_BY_KEY = {
    "asvspoof2019_la": "single_chunk",
    "commonvoice_ko": "single_chunk",
    "musan/speech": "sequential_drop",
    "musan/music": "sequential_drop",
    "musan/noise": "noise_conditional",
    "fakemusiccaps": "single_chunk",
    "vctk": "single_chunk",
    "dfadd": "single_chunk",
    "wavefake": "single_chunk",
}

DFADD_MODELS = ["GradTTS", "MatchaTTS", "NaturalSpeech2", "PflowTTS", "StyleTTS2"]
OFFICIAL_SPLIT_MAP = {"train": "train", "valid": "val", "test": "test"}

# ===== split =====
SPLIT_RATIOS = (0.8, 0.1, 0.1)
SPLIT_SEED = 42
SPLIT_NAMES = ["train", "val", "test"]

# ===== 실행 =====
NUM_WORKERS = max(1, (os.cpu_count() or 4) - 1)
MAX_FILES_PER_DATASET = None

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".ogg")

COLUMN_ORDER = [
    "chunk_id", "filepath", "dataset", "split", "orig_split", "category", "source_model",
    "VP", "MP", "VF", "MF", "FF",
    "chunk_policy",
    "chunk_index", "n_chunks_in_file", "is_padded", "pad_seconds", "is_truncated",
    "orig_duration_sec", "orig_sr", "orig_channels",
    "measured_lufs", "target_lufs", "gain_applied_db", "peak_after_norm", "loudness_norm_applied",
    "group_id", "source_path",
]

PARTIAL_CSV = META_OUT / "chunks_partial.csv"

# ===== 통계 추적 =====
STATS = {
    "total_files": 0,
    "total_chunks": 0,
    "filtered_chunks": 0,  # 4초 미만으로 필터링된 청크
    "by_dataset": {},
}


def probe_audio_info(path: Path) -> Dict[str, Optional[float]]:
    """원본 파일 정보 조회"""
    try:
        info = sf.info(str(path))
        duration = info.frames / info.samplerate if info.samplerate else None
        return {"samplerate": info.samplerate, "channels": info.channels, "duration_sec": duration}
    except Exception:
        return {"samplerate": None, "channels": None, "duration_sec": None}


def load_mono_16k(path: Path, target_sr: int = TARGET_SR) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """오디오 로드"""
    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
        return y.astype(np.float32), sr
    except Exception as e:
        return None, str(e)


def normalize_loudness(
    y: np.ndarray,
    sr: int,
    target_lufs: float = TARGET_LUFS,
    peak_ceiling: float = PEAK_CEILING
) -> Tuple[np.ndarray, Optional[float], float, float, bool]:
    """라우드니스 정규화"""
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


def chunk_audio(
    y: np.ndarray,
    policy: str,
    chunk_samples: int = CHUNK_SAMPLES
) -> List[Tuple[np.ndarray, bool, float, bool]]:
    """청크 분할 (4초 미만 필터링 추가)"""

    def _single(y_):
        n = len(y_)
        if n == chunk_samples:
            # 정확히 10초 - 유효 길이 체크
            return [(y_, False, 0.0, False)] if n >= MIN_CHUNK_SAMPLES else []
        if n > chunk_samples:
            # 10초보다 길면 앞 10초 사용
            return [(y_[:chunk_samples], False, 0.0, True)] if chunk_samples >= MIN_CHUNK_SAMPLES else []
        # 10초보다 짧음 - 4초 이상이면 zero-pad, 4초 미만이면 필터링
        if n < MIN_CHUNK_SAMPLES:
            return []  # 필터링
        pad_width = chunk_samples - n
        padded = np.pad(y_, (0, pad_width), mode="constant")
        return [(padded, True, pad_width / TARGET_SR, False)]

    def _sequential_drop(y_):
        n_full = len(y_) // chunk_samples
        if n_full == 0:
            return _single(y_)
        chunks = []
        for i in range(n_full):
            chunk = y_[i * chunk_samples:(i + 1) * chunk_samples]
            # 4초 이상이면 포함
            if len(chunk) >= MIN_CHUNK_SAMPLES:
                chunks.append((chunk, False, 0.0, False))
        return chunks if chunks else _single(y_)

    if policy == "single_chunk":
        return _single(y)
    if policy == "sequential_drop":
        return _sequential_drop(y)
    if policy == "noise_conditional":
        return _sequential_drop(y) if len(y) >= chunk_samples else _single(y)
    raise ValueError(f"unknown chunk_policy: {policy!r}")


def _process_file_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """파일 처리 (병렬 처리용)"""
    in_path = job["in_path"]
    out_base_dir = Path(job["out_base_dir"])
    base_name = job["base_name"]
    label = job["label"]
    chunk_policy = label["chunk_policy"]
    split = label["split"]

    try:
        info = probe_audio_info(in_path)
        y, sr = load_mono_16k(in_path)
        if y is None or len(y) == 0:
            return {"rows": [], "error": f"{in_path}: load failed ({sr})"}

        y_norm, measured_lufs, gain_db, peak, norm_ok = normalize_loudness(y, sr)
        chunks = chunk_audio(y_norm, chunk_policy)

        if not chunks:
            # 4초 미만이라 모두 필터링됨
            return {"rows": [], "error": None, "filtered": True}

        out_dir = out_base_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for idx, (arr, is_padded, pad_seconds, is_truncated) in enumerate(chunks):
            out_path = out_dir / f"{base_name}_chunk{idx:04d}.wav"
            arr = np.clip(arr, -1.0, 1.0)
            sf.write(str(out_path), arr, TARGET_SR, subtype="PCM_16")

            rel_path = out_path.relative_to(AUDIO_OUT)
            chunk_id = str(rel_path.with_suffix("")).replace("/", "__")

            row = dict(label)
            row.update(
                chunk_id=chunk_id,
                filepath=str(rel_path),
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
        return {"rows": rows, "error": None, "filtered": False}
    except Exception as e:
        return {"rows": [], "error": f"{in_path}: {e}", "filtered": False}


def run_jobs(jobs: List[Dict], desc: str, num_workers: int = NUM_WORKERS) -> Tuple[List[Dict], int]:
    """병렬 처리"""
    all_rows = []
    errors = []
    filtered_count = 0

    if num_workers <= 1:
        results = (_process_file_job(j) for j in jobs)
        for res in tqdm(results, total=len(jobs), desc=desc):
            all_rows.extend(res["rows"])
            if res.get("filtered"):
                filtered_count += 1
            if res["error"]:
                errors.append(res["error"])
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            for res in tqdm(ex.map(_process_file_job, jobs, chunksize=8), total=len(jobs), desc=desc):
                all_rows.extend(res["rows"])
                if res.get("filtered"):
                    filtered_count += 1
                if res["error"]:
                    errors.append(res["error"])

    if errors:
        print(f"[{desc}] {len(errors)}개 실패, 예시:", errors[:3])
    return all_rows, filtered_count


def load_done_source_paths() -> set:
    """이미 처리된 파일 목록"""
    if not PARTIAL_CSV.exists():
        return set()
    return set(pd.read_csv(PARTIAL_CSV, usecols=["source_path"])["source_path"])


def append_rows_checkpoint(rows: List[Dict]):
    """체크포인트 저장"""
    if not rows:
        return
    chunk_df = pd.DataFrame(rows)[COLUMN_ORDER]
    chunk_df.to_csv(PARTIAL_CSV, mode="a", header=not PARTIAL_CSV.exists(), index=False)


_WAVEFAKE_CONTENT_RE = re.compile(r"^(LJ\d+-\d+|BASIC5000_\d+)")


def wavefake_content_id(path: Path, model_dir_name: str) -> str:
    """서로 다른 vocoder가 같은 원문을 합성한 경우 동일한 group으로 묶어
    같은 문장이 train/val/test에 걸쳐 나뉘는 것(leakage)을 방지"""
    m = _WAVEFAKE_CONTENT_RE.match(path.stem)
    if m:
        return m.group(1)
    return f"{model_dir_name}::{path.stem}"


def grouped_split(groups: pd.Series, ratios=SPLIT_RATIOS, seed=SPLIT_SEED) -> Dict[str, str]:
    """그룹 단위 분할"""
    sizes = groups.value_counts()
    order = sizes.index.to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(order)

    total = sizes.sum()
    names = list(SPLIT_NAMES)
    targets = {n: r * total for n, r in zip(names, ratios)}
    assigned = {n: 0 for n in names}

    mapping = {}
    for g in order:
        need = {n: (assigned[n] / targets[n] if targets[n] > 0 else 1.0) for n in names}
        choice = min(names, key=lambda n: need[n])
        mapping[g] = choice
        assigned[choice] += sizes[g]
    return mapping


print("=" * 60)
print("v3 오디오 전처리 파이프라인 시작")
print(f"TARGET_SR={TARGET_SR}, CHUNK_SEC={CHUNK_SEC}, MIN_CHUNK_SEC={MIN_CHUNK_SEC}")
print(f"NUM_WORKERS={NUM_WORKERS}, MAX_FILES={MAX_FILES_PER_DATASET}")
print(f"OUT_ROOT={OUT_ROOT}")
if PARTIAL_CSV.exists():
    n_done = sum(1 for _ in open(PARTIAL_CSV)) - 1
    print(f"체크포인트: {PARTIAL_CSV} ({n_done}개 chunk 기록됨)")
print("=" * 60)

done = load_done_source_paths()
all_rows = []

# ===== ASVspoof =====
print("\n[1/9] ASVspoof 2019 LA 처리 중...")
ASV_ROOT = RAW_ROOT / "asvspoof2019_la" / "extracted" / "LA"
ASV_PROTOCOLS = ASV_ROOT / "ASVspoof2019_LA_cm_protocols"

ASV_SPLITS = {
    "train": ("ASVspoof2019_LA_train", "ASVspoof2019.LA.cm.train.trn.txt", "train"),
    "dev": ("ASVspoof2019_LA_dev", "ASVspoof2019.LA.cm.dev.trl.txt", "val"),
    "eval": ("ASVspoof2019_LA_eval", "ASVspoof2019.LA.cm.eval.trl.txt", "test"),
}

asv_skipped = 0
asv_filtered = 0
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
            in_path=str(f),
            out_base_dir=str(AUDIO_OUT / "asvspoof2019_la" / orig_split),
            base_name=f.stem, label=label,
        ))

    new_rows, filtered = run_jobs(jobs, desc=f"ASVspoof/{orig_split}")
    asv_filtered += filtered
    append_rows_checkpoint(new_rows)
    all_rows.extend(new_rows)

print(f"ASVspoof: {len(all_rows)}개 chunk, 이어서 건너뜀: {asv_skipped}, 필터됨: {asv_filtered}")
STATS["by_dataset"]["asvspoof2019_la"] = {"chunks": len(all_rows), "filtered": asv_filtered}

# ===== Common Voice =====
print("\n[2/9] Common Voice Korean 처리 중...")
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

cv_files = sorted((CV_ROOT / "clips").glob("*.mp3"))[:MAX_FILES_PER_DATASET]
cv_skipped = 0
cv_candidates = []  # (f, client_id, official_split)
for f in cv_files:
    if str(f) in done:
        cv_skipped += 1
        continue
    client_id = client_lookup.get(f.name, f.stem)
    official_split = official_split_map.get(f.name)
    cv_candidates.append((f, client_id, official_split))

# 공식 split이 없는 파일은 처리 전에 미리 group 단위로 split을 배정해야
# out_base_dir / split 경로 조합 시 split=None으로 인한 오류가 나지 않음
cv_missing_group_ids = pd.Series([
    f"commonvoice_ko::{cid}" for _, cid, osplit in cv_candidates if osplit is None
])
cv_extra_split_map = (
    grouped_split(cv_missing_group_ids, seed=SPLIT_SEED + 1) if len(cv_missing_group_ids) else {}
)

cv_jobs = []
for f, client_id, official_split in cv_candidates:
    group_id = f"commonvoice_ko::{client_id}"
    split = official_split or cv_extra_split_map.get(group_id, "train")
    label = dict(
        dataset="commonvoice_ko", VP=1, MP=0, VF=0, MF=-1, FF=0,
        source_model=None, category=None,
        chunk_policy=CHUNK_POLICY_BY_KEY["commonvoice_ko"],
        orig_split=("cv_official" if official_split else "cv_grouped"), split=split,
        group_id=group_id, source_path=str(f),
    )
    cv_jobs.append(dict(
        in_path=str(f), out_base_dir=str(AUDIO_OUT / "commonvoice_ko"),
        base_name=f.stem, label=label,
    ))

cv_rows, cv_filtered = run_jobs(cv_jobs, desc="Common Voice")
append_rows_checkpoint(cv_rows)
all_rows.extend(cv_rows)
print(f"Common Voice: {len(cv_rows)}개 chunk, 이어서 건너뜀: {cv_skipped}, 필터됨: {cv_filtered}")
STATS["by_dataset"]["commonvoice_ko"] = {"chunks": len(cv_rows), "filtered": cv_filtered}

# ===== MUSAN =====
print("\n[3/9] MUSAN 처리 중...")
MUSAN_ROOT = RAW_ROOT / "musan" / "extracted" / "musan"
MUSAN_RULES = {
    "speech": dict(VP=1, MP=0, VF=0, MF=-1, FF=0, chunk_policy=CHUNK_POLICY_BY_KEY["musan/speech"]),
    "music": dict(VP=0, MP=1, VF=-1, MF=0, FF=0, chunk_policy=CHUNK_POLICY_BY_KEY["musan/music"]),
    "noise": dict(VP=0, MP=0, VF=-1, MF=-1, FF=0, chunk_policy=CHUNK_POLICY_BY_KEY["musan/noise"]),
}

musan_skipped = 0
musan_candidates = []  # (f, category)
for category, rule in MUSAN_RULES.items():
    files = sorted((MUSAN_ROOT / category).rglob("*.wav"))[:MAX_FILES_PER_DATASET]
    for f in files:
        if str(f) in done:
            musan_skipped += 1
            continue
        musan_candidates.append((f, category))

# MUSAN은 전체 데이터셋 단위로 group split을 미리 계산 (원본 stem 기준, 카테고리 무관)
musan_group_ids = pd.Series([f"musan::{f.stem}" for f, _ in musan_candidates])
musan_split_map = grouped_split(musan_group_ids, seed=SPLIT_SEED + 2) if len(musan_group_ids) else {}

musan_filtered = 0
musan_rows_total = []
for category, rule in MUSAN_RULES.items():
    jobs = []
    for f, cat in musan_candidates:
        if cat != category:
            continue
        subsource = f.parent.name
        group_id = f"musan::{f.stem}"
        split = musan_split_map.get(group_id, "train")
        label = dict(
            dataset="musan", **rule,
            source_model=None, category=f"{category}/{subsource}",
            orig_split=None, split=split,
            group_id=group_id, source_path=str(f),
        )
        jobs.append(dict(
            in_path=str(f), out_base_dir=str(AUDIO_OUT / "musan" / category / subsource),
            base_name=f.stem, label=label,
        ))
    new_rows, filtered = run_jobs(jobs, desc=f"MUSAN/{category}")
    musan_filtered += filtered
    append_rows_checkpoint(new_rows)
    all_rows.extend(new_rows)
    musan_rows_total.extend(new_rows)

print(f"MUSAN: {len(musan_rows_total)}개 chunk, 이어서 건너뜀: {musan_skipped}, 필터됨: {musan_filtered}")

# ===== FakeMusicCaps =====
print("\n[4/9] FakeMusicCaps 처리 중...")
FMC_ROOT = RAW_ROOT / "fakemusiccaps" / "processed"

fmc_skipped = 0
fmc_candidates = []  # (f, model_dir)
for model_dir in sorted(d for d in FMC_ROOT.iterdir() if d.is_dir() and d.name != "__MACOSX"):
    files = sorted(model_dir.glob("*.wav"))[:MAX_FILES_PER_DATASET]
    for f in files:
        if str(f) in done:
            fmc_skipped += 1
            continue
        fmc_candidates.append((f, model_dir))

fmc_group_ids = pd.Series([f"fakemusiccaps::{f.stem}" for f, _ in fmc_candidates])
fmc_split_map = grouped_split(fmc_group_ids, seed=SPLIT_SEED + 3) if len(fmc_group_ids) else {}

fmc_jobs = []
for f, model_dir in fmc_candidates:
    group_id = f"fakemusiccaps::{f.stem}"
    split = fmc_split_map.get(group_id, "train")
    label = dict(
        dataset="fakemusiccaps", VP=0, MP=1, VF=-1, MF=1, FF=1,
        source_model=model_dir.name, category=None,
        chunk_policy=CHUNK_POLICY_BY_KEY["fakemusiccaps"],
        orig_split=None, split=split,
        group_id=group_id, source_path=str(f),
    )
    fmc_jobs.append(dict(
        in_path=str(f), out_base_dir=str(AUDIO_OUT / "fakemusiccaps" / model_dir.name),
        base_name=f.stem, label=label,
    ))

fmc_rows, fmc_filtered = run_jobs(fmc_jobs, desc="FakeMusicCaps")
append_rows_checkpoint(fmc_rows)
all_rows.extend(fmc_rows)
print(f"FakeMusicCaps: {len(fmc_rows)}개 chunk, 이어서 건너뜀: {fmc_skipped}, 필터됨: {fmc_filtered}")

# ===== VCTK (real bonafide, DFADD의 실제 음성 짝) =====
print("\n[5/9] VCTK (bonafide) 처리 중...")
VCTK_ROOT = RAW_ROOT / "dfadd" / "DATASET_VCTK_BONAFIDE"

vctk_skipped = 0
vctk_jobs = []
if VCTK_ROOT.exists():
    for folder, mapped_split in OFFICIAL_SPLIT_MAP.items():
        files = sorted((VCTK_ROOT / folder).glob("*.wav"))[:MAX_FILES_PER_DATASET]
        for f in files:
            if str(f) in done:
                vctk_skipped += 1
                continue
            speaker_id = f.stem.split("_")[0]
            label = dict(
                dataset="vctk", VP=1, MP=0, VF=0, MF=-1, FF=0,
                source_model=None, category="vctk_bonafide",
                chunk_policy=CHUNK_POLICY_BY_KEY["vctk"],
                orig_split=folder, split=mapped_split,
                group_id=f"vctk::{speaker_id}", source_path=str(f),
            )
            vctk_jobs.append(dict(
                in_path=str(f), out_base_dir=str(AUDIO_OUT / "vctk"),
                base_name=f.stem, label=label,
            ))
else:
    print(f"⚠️ VCTK 경로 없음: {VCTK_ROOT}")

vctk_rows, vctk_filtered = run_jobs(vctk_jobs, desc="VCTK")
append_rows_checkpoint(vctk_rows)
all_rows.extend(vctk_rows)
print(f"VCTK: {len(vctk_rows)}개 chunk, 이어서 건너뜀: {vctk_skipped}, 필터됨: {vctk_filtered}")

# ===== DFADD (fake TTS, 5개 모델) =====
print("\n[6/9] DFADD (fake TTS) 처리 중...")
DFADD_ROOT = RAW_ROOT / "dfadd"

dfadd_skipped = 0
dfadd_jobs = []
for model_name in DFADD_MODELS:
    model_root = DFADD_ROOT / f"DATASET_{model_name}"
    if not model_root.exists():
        print(f"⚠️ DFADD 모델 경로 없음: {model_root}")
        continue
    for folder, mapped_split in OFFICIAL_SPLIT_MAP.items():
        files = sorted((model_root / folder).glob("*.flac"))[:MAX_FILES_PER_DATASET]
        for f in files:
            if str(f) in done:
                dfadd_skipped += 1
                continue
            speaker_id = f.stem.split("_")[0]
            label = dict(
                dataset="dfadd", VP=1, MP=0, VF=1, MF=-1, FF=1,
                source_model=model_name, category=f"dfadd/{model_name}",
                chunk_policy=CHUNK_POLICY_BY_KEY["dfadd"],
                orig_split=folder, split=mapped_split,
                group_id=f"dfadd::{speaker_id}", source_path=str(f),
            )
            dfadd_jobs.append(dict(
                in_path=str(f), out_base_dir=str(AUDIO_OUT / "dfadd" / model_name),
                base_name=f.stem, label=label,
            ))

dfadd_rows, dfadd_filtered = run_jobs(dfadd_jobs, desc="DFADD")
append_rows_checkpoint(dfadd_rows)
all_rows.extend(dfadd_rows)
print(f"DFADD: {len(dfadd_rows)}개 chunk, 이어서 건너뜀: {dfadd_skipped}, 필터됨: {dfadd_filtered}")

# ===== WaveFake (fake vocoder, 공식 split 없음 -> 내용 기준 group split) =====
print("\n[7/9] WaveFake (vocoder fake) 처리 중...")
WAVEFAKE_ROOT = RAW_ROOT / "wavefake" / "generated_audio"

wavefake_skipped = 0
wavefake_candidates = []  # (f, model_dir_name, content_id)
if WAVEFAKE_ROOT.exists():
    for model_dir in sorted(d for d in WAVEFAKE_ROOT.iterdir() if d.is_dir()):
        files = sorted(model_dir.glob("*.wav"))[:MAX_FILES_PER_DATASET]
        for f in files:
            if str(f) in done:
                wavefake_skipped += 1
                continue
            content_id = wavefake_content_id(f, model_dir.name)
            wavefake_candidates.append((f, model_dir.name, content_id))
else:
    print(f"⚠️ WaveFake 경로 없음: {WAVEFAKE_ROOT}")

# 서로 다른 vocoder가 같은 원문을 합성한 경우 group_id가 같아지도록 하여
# 동일 문장이 train/val/test로 나뉘어 새는 것을 방지 (musan/fmc와 동일한 패턴)
wavefake_group_ids = pd.Series([f"wavefake::{cid}" for _, _, cid in wavefake_candidates])
wavefake_split_map = (
    grouped_split(wavefake_group_ids, seed=SPLIT_SEED + 4) if len(wavefake_group_ids) else {}
)

wavefake_jobs = []
for f, model_name, content_id in wavefake_candidates:
    group_id = f"wavefake::{content_id}"
    split = wavefake_split_map.get(group_id, "train")
    label = dict(
        dataset="wavefake", VP=1, MP=0, VF=1, MF=-1, FF=1,
        source_model=model_name, category=f"wavefake/{model_name}",
        chunk_policy=CHUNK_POLICY_BY_KEY["wavefake"],
        orig_split=None, split=split,
        group_id=group_id, source_path=str(f),
    )
    wavefake_jobs.append(dict(
        in_path=str(f), out_base_dir=str(AUDIO_OUT / "wavefake" / model_name),
        base_name=f.stem, label=label,
    ))

wavefake_rows, wavefake_filtered = run_jobs(wavefake_jobs, desc="WaveFake")
append_rows_checkpoint(wavefake_rows)
all_rows.extend(wavefake_rows)
print(f"WaveFake: {len(wavefake_rows)}개 chunk, 이어서 건너뜀: {wavefake_skipped}, 필터됨: {wavefake_filtered}")

# ===== split 검증 =====
# v3에서는 각 데이터셋 처리 시점에 group 단위로 split을 미리 배정하므로
# (out_base_dir / split 경로 조합 시 None이 들어가는 것을 방지하기 위함)
# 여기서는 최종 검증만 수행
print("\n[8/9] split 배정 검증 중...")
df = pd.read_csv(PARTIAL_CSV)
print(f"전체 chunk 수 (누적): {len(df)}")

n_missing = df["split"].isna().sum()
if n_missing:
    print(f"⚠️ split 미배정 행 {n_missing}개 발견 — train으로 대체")
    df["split"] = df["split"].fillna("train")

assert df["split"].isna().sum() == 0, "split이 배정되지 않은 행이 남아있음"
df = df[COLUMN_ORDER]

print("\n전체 dataset/split 분포:")
print(pd.crosstab(df["dataset"], df["split"]))

# ===== 메타데이터 저장 =====
print("\n[9/9] 메타데이터 저장 중...")
df.to_csv(META_OUT / "chunks.csv", index=False)

for split_name in SPLIT_NAMES:
    df[df["split"] == split_name].to_csv(META_OUT / f"chunks_{split_name}.csv", index=False)

voice_df = df[df["VP"] == 1]
music_df = df[df["MP"] == 1]
voice_df.to_csv(META_OUT / "voice_pool.csv", index=False)
music_df.to_csv(META_OUT / "music_pool.csv", index=False)

print(f"\nVoice pool: {len(voice_df)}개 청크")
print(f"Music pool: {len(music_df)}개 청크")

# ===== 최종 통계 저장 =====
STATS["total_files"] = int(df.groupby("dataset")["source_path"].nunique().sum())
STATS["total_chunks"] = len(df)
STATS["filtered_chunks"] = (
    asv_filtered + cv_filtered + musan_filtered + fmc_filtered
    + vctk_filtered + dfadd_filtered + wavefake_filtered
)
print("\n데이터셋별 chunk 수:")
print(df.groupby("dataset").size())

config = {
    "version": "v3",
    "target_sr": TARGET_SR,
    "chunk_sec": CHUNK_SEC,
    "min_chunk_sec": MIN_CHUNK_SEC,
    "target_lufs": TARGET_LUFS,
    "peak_ceiling_db": PEAK_CEILING_DB,
    "split_ratios": dict(zip(SPLIT_NAMES, SPLIT_RATIOS)),
    "split_seed": SPLIT_SEED,
    "max_files_per_dataset": MAX_FILES_PER_DATASET,
    "chunk_policy_by_dataset": CHUNK_POLICY_BY_KEY,
    "num_source_files": STATS["total_files"],
    "num_chunks": STATS["total_chunks"],
    "num_filtered_chunks": STATS["filtered_chunks"],
}

try:
    with open(META_OUT / "preprocessing_config_v3.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
except TypeError as e:
    print(f"⚠️ JSON 저장 중 오류 (안전하게 처리됨): {e}")
    # 문제가 되는 부분만 제거하고 다시 시도
    config.pop("label_distribution_summary", None)
    with open(META_OUT / "preprocessing_config_v3.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print(f"전처리 완료!")
print(f"총 원본 파일: {STATS['total_files']}")
print(f"생성된 청크: {STATS['total_chunks']}")
print(f"필터링된 청크 (4초 미만): {STATS['filtered_chunks']}")
print(f"OUT_ROOT: {OUT_ROOT}")
print("=" * 60)
