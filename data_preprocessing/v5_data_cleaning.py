#!/usr/bin/env python3
"""
v5: data_mixed_v4를 기반으로, 라벨 있는 8개 그룹(noise_only 제외)마다 일부 샘플을
배경잡음이 섞인 버전으로 "교체"한다. (전체 개수는 v4와 동일하게 유지 — 추가가 아니라
일부 대체)

- 대상: voice_only / voice_fake_only / music_only / music_fake_only / voice_music /
  voice_music_fake / voice_fake_music / voice_fake_music_fake (그룹당 25%)
  noise_only는 이미 전부 잡음이라 대상에서 제외.
- 범위: train + val + test 전부 (실전 배포 환경처럼 잡음 섞인 조건에서의 성능도
  평가에 반영하기로 함 — 사용자 확인됨).
- 라벨(VP/MP/VF/MF/FF)은 바꾸지 않는다 — 배경잡음을 더하는 건 그 안의 음성/음악이
  진짜인지 가짜인지와 무관하기 때문. 대신 `noise_augmented`(bool)/`noise_method`
  컬럼을 추가해서 어떤 샘플이 잡음 처리됐는지 라벨링/추적 가능하게 한다.
- 잡음 소스: MUSAN noise(930개, free-sound+sound-bible). RIRS_NOISES의
  pointsource_noises는 확인해보니 MUSAN noise/free-sound와 843개 전부 중복이라
  새로운 다양성을 안 주므로 별도로 쓰지 않음. 대신 RIRS_NOISES의 simulated_rirs
  (잔향, 완전히 다른 자원)를 절반 확률로 같이 섞어서 "잡음+잔향" 현실감을 더함.
- 교체 안 되는 75%는 v4 오디오를 그대로 복사(재인코딩 없이 shutil.copy2)해서
  data_mixed_v5가 완전히 독립적인 단일 폴더가 되도록 한다.

Usage:
  python build_mixed_dataset_v5.py
  또는 tmux에서:
  tmux new-session -d -s mixed_v5 'cd /cluster/workspace/deepvoice && /cluster/workspace/envs/deepvoice/bin/python yusung/build_mixed_dataset_v5.py'
"""
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from augment_common_v3 import (
    TARGET_SR, fit_length, resample_if_needed, add_reverb, add_background, clip_normalize,
)

V4_MIXED = Path("/cluster/workspace/deepvoice/yusung/data_mixed_v4")
V4_AUDIO = V4_MIXED / "audio"
V4_META = V4_MIXED / "metadata"

V5_MIXED = Path("/cluster/workspace/deepvoice/yusung/data_mixed_v5")
V5_AUDIO = V5_MIXED / "audio"
V5_META = V5_MIXED / "metadata"
V5_AUDIO.mkdir(parents=True, exist_ok=True)
V5_META.mkdir(parents=True, exist_ok=True)

RAW_ROOT = Path("/cluster/workspace/deepvoice/data/raw")
MUSAN_NOISE_DIR = RAW_ROOT / "musan" / "extracted" / "musan" / "noise"
RIR_DIR = RAW_ROOT / "rirs_noises" / "extracted" / "RIRS_NOISES" / "simulated_rirs"

CHUNK_SAMPLES = TARGET_SR * 10
NOISE_RATIO = 0.25  # 그룹당 잡음 교체 비율
REVERB_PROB = 0.5   # 잡음 섞을 때 잔향도 같이 섞을 확률
SEED = 777

EXCLUDED_GROUP = "noise_only"


def load_musan_noise_paths():
    return sorted(MUSAN_NOISE_DIR.rglob("*.wav"))


def load_rir_paths():
    return sorted(RIR_DIR.rglob("*.wav"))


def apply_noise(y: np.ndarray, sr: int, rng: random.Random, noise_paths, rir_paths) -> tuple:
    noise_path = rng.choice(noise_paths)
    noise, noise_sr = sf.read(str(noise_path))
    if noise.ndim > 1:
        noise = noise.mean(axis=1)
    noise = resample_if_needed(noise, noise_sr, sr)
    snr = rng.uniform(5, 20)
    y = add_background(y, noise, snr, rng)
    method = f"bg({noise_path.stem},{snr:.0f}dB)"

    if rir_paths and rng.random() < REVERB_PROB:
        rir_path = rng.choice(rir_paths)
        rir, rir_sr = sf.read(str(rir_path))
        if rir.ndim > 1:
            rir = rir.mean(axis=1)
        rir = resample_if_needed(rir, rir_sr, sr)
        y = add_reverb(y, rir)
        method += f"+reverb({rir_path.stem})"

    y = fit_length(y, CHUNK_SAMPLES)
    y = clip_normalize(y)
    return y, method


def main():
    print("=" * 60)
    print("v5: data_mixed_v4 기반 배경잡음 교체 증강 시작")
    print("=" * 60)

    mixed = pd.read_csv(V4_META / "mixed.csv", low_memory=False)
    mixed["group"] = mixed["mixed_id"].str.extract(r"^([a-z_]+)_(?:train|val|test)_(?:\d+|sonics_\d+)$")
    mixed["noise_augmented"] = False
    mixed["noise_method"] = None

    noise_paths = load_musan_noise_paths()
    rir_paths = load_rir_paths()
    print(f"MUSAN noise {len(noise_paths)}개, RIR(잔향) {len(rir_paths)}개 로드됨")

    rng_py = random.Random(SEED)
    rng_np = np.random.default_rng(SEED)

    noise_target_idx = set()
    for (group, split), sub in mixed.groupby(["group", "split"]):
        if group == EXCLUDED_GROUP:
            continue
        n = round(len(sub) * NOISE_RATIO)
        if n <= 0:
            continue
        chosen = rng_np.choice(sub.index.to_numpy(), size=n, replace=False)
        noise_target_idx.update(chosen.tolist())

    print(f"\n잡음 교체 대상: {len(noise_target_idx)}개 / 전체 {len(mixed)}개 "
          f"({len(noise_target_idx)/len(mixed)*100:.1f}%)")

    successful = 0
    failed = 0
    for idx, row in tqdm(mixed.iterrows(), total=len(mixed), desc="v5 오디오"):
        src_path = V4_AUDIO / row["split"] / f"{row['mixed_id']}.wav"
        dst_path = V5_AUDIO / row["split"] / f"{row['mixed_id']}.wav"
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        if idx not in noise_target_idx:
            try:
                shutil.copy2(src_path, dst_path)
                successful += 1
            except Exception as e:
                print(f"복사 실패: {e}")
                failed += 1
            continue

        try:
            y, sr = sf.read(str(src_path))
            if y.ndim > 1:
                y = y.mean(axis=1)
            y_noisy, method = apply_noise(y.astype(np.float32), sr, rng_py, noise_paths, rir_paths)
            sf.write(str(dst_path), y_noisy, TARGET_SR, subtype="PCM_16")
            mixed.at[idx, "noise_augmented"] = True
            mixed.at[idx, "noise_method"] = method
            successful += 1
        except Exception as e:
            print(f"잡음 합성 실패: {e}")
            failed += 1

    print(f"\n성공: {successful}, 실패: {failed}")

    mixed_out = mixed.drop(columns=["group"])
    mixed_out.to_csv(V5_META / "mixed.csv", index=False)
    for split in ["train", "val", "test"]:
        mixed_out[mixed_out["split"] == split].to_csv(V5_META / f"mixed_{split}.csv", index=False)

    print("\n" + "=" * 60)
    print("v5 배경잡음 교체 증강 완료!")
    print(f"총 샘플: {len(mixed_out)}, 잡음 교체된 샘플: {mixed_out['noise_augmented'].sum()}")
    print(f"MIXED_OUT: {V5_MIXED}")
    print("=" * 60)


if __name__ == "__main__":
    main()
