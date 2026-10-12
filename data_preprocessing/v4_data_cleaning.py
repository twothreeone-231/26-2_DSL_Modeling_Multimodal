#!/usr/bin/env python3
"""
v4: data_processed_v3의 chunks.csv에서 "보컬이 섞인 MUSAN music" 청크를 제거해
data_processed_v4/metadata/{chunks,chunks_train,voice_pool,music_pool}.csv를 만든다.

배경: MUSAN music의 5개 서브폴더(fma/jamendo/rfm/fma-western-art/hd-classical) 중
fma/jamendo/fma-western-art는 자체 ANNOTATIONS 파일에 보컬 유무(Y/N)가 원본 단위로
기록돼 있는데, v3는 이걸 무시하고 전부 "순수 음악"(VP=0)으로 썼다. 실측 결과
musan/music 청크 17,839개 중 5,958개(33.4%)가 보컬 섞인 원본에서 나온 것이었다
(real 4,981개 + 그 원본에서 파생된 augmented 977개 포함).

이 스크립트는 원본 wav 파일을 지우지 않는다(data_processed_v3는 그대로 보존) —
v4용 메타데이터에서 해당 행만 제외하고, 물리적 오디오는 v3와 계속 공유한다
(AUDIO_ROOT는 data_processed_v3/audio 그대로).

Usage:
  python build_processed_v4_pools.py
"""
import re
from pathlib import Path

import pandas as pd

V3_META = Path("/data1/dsl07/deepvoice/yusung/data_processed_v3/metadata")
V4_META = Path("/data1/dsl07/deepvoice/yusung/data_processed_v4/metadata")
V4_META.mkdir(parents=True, exist_ok=True)

MUSAN_MUSIC_ROOT = Path("/data1/dsl07/deepvoice/data/raw/musan/extracted/musan/music")
ANNOTATED_SUBDIRS = ["fma", "jamendo", "rfm", "fma-western-art", "hd-classical"]

_STEM_RE = re.compile(r"__(music-[a-z-]+-\d+)_chunk")


def load_vocal_stems() -> set:
    """MUSAN 공식 ANNOTATIONS에서 보컬(Y) 표시된 원본 파일 stem 집합을 만든다."""
    vocal_stems = set()
    for d in ANNOTATED_SUBDIRS:
        path = MUSAN_MUSIC_ROOT / d / "ANNOTATIONS"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 3 and parts[2] == "Y":
                    vocal_stems.add(parts[0])
    return vocal_stems


def resolve_orig_stem(row: pd.Series) -> str:
    """실제(real) 행은 자기 chunk_id에서, 증강(augmented) 행은 source_chunk_id에서
    원본 MUSAN music 파일 stem을 역추적한다 (증강도 오염된 원본에서 파생됐으면 같이 제외)."""
    cid = row["source_chunk_id"] if row.get("is_augmented") else row["chunk_id"]
    m = _STEM_RE.search(str(cid))
    return m.group(1) if m else None


def main():
    print("=" * 60)
    print("v4 data_processed 풀 생성 (MUSAN music 보컬 오염 제거)")
    print("=" * 60)

    vocal_stems = load_vocal_stems()
    print(f"보컬(Y) 표시된 MUSAN music 원본: {len(vocal_stems)}개")

    df = pd.read_csv(V3_META / "chunks.csv", low_memory=False)
    print(f"v3 전체 청크: {len(df)}개")

    is_music = (df["dataset"] == "musan") & (df["category"].astype(str).str.startswith("music"))
    df.loc[is_music, "_orig_stem"] = df.loc[is_music].apply(resolve_orig_stem, axis=1)
    contaminated = is_music & df["_orig_stem"].isin(vocal_stems)

    print(f"musan/music 전체: {is_music.sum()}개, 오염되어 제거: {contaminated.sum()}개")
    print(contaminated[contaminated].index.size and df.loc[contaminated, "split"].value_counts())

    df_clean = df.loc[~contaminated].drop(columns=["_orig_stem"]).reset_index(drop=True)
    print(f"제거 후 남은 청크: {len(df_clean)}개")

    df_clean.to_csv(V4_META / "chunks.csv", index=False)
    df_clean[df_clean["split"] == "train"].to_csv(V4_META / "chunks_train.csv", index=False)
    df_clean[df_clean["split"] == "val"].to_csv(V4_META / "chunks_val.csv", index=False)
    df_clean[df_clean["split"] == "test"].to_csv(V4_META / "chunks_test.csv", index=False)

    voice_df = df_clean[df_clean["VP"] == 1]
    music_df = df_clean[df_clean["MP"] == 1]
    voice_df.to_csv(V4_META / "voice_pool.csv", index=False)
    music_df.to_csv(V4_META / "music_pool.csv", index=False)

    print(f"\nvoice_pool: {len(voice_df)}개, music_pool: {len(music_df)}개")
    print("데이터셋별:")
    print(df_clean["dataset"].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()
