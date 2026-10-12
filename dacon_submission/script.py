#!/usr/bin/env python3
"""DACON code-submission entry point: AudioMosaic + BS PolarFormer (voice/
music presence) + lofcz/ai-music-detector (music deepfake, uncalibrated --
see src/music_fake_lofcz.py and SUBMISSION_LOG.md) + DF-Arena 1B (voice
deepfake). BS PolarFormer separation only runs when
music_routing.decide_processing_plan says so (neither component's zone is
confidently absent) -- see src/music_routing.py's module docstring. A prior
submission that always separated already failed with a 60-minute execution
timeout (measured ~4s/file on GPU, dominated by separation, vs ~0.11s/file
when separation is skipped). See jaehyun/scripts/generate_submission.py
(dev-time equivalent) and jaehyun/README.md for the full pipeline explanation.
"""
import csv
import os
import sys
import time
from pathlib import Path

# --- offline mode: no internet on the eval server -----------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True

BASE_DIR = Path(__file__).resolve().parent
os.environ["JAEHYUN_HF_CACHE_DIR"] = str(BASE_DIR / "model" / "hf_cache")
os.environ["JAEHYUN_ONTOLOGY_CSV"] = str(BASE_DIR / "model" / "audioset_ontology" / "class_labels_indices.csv")
os.environ["JAEHYUN_CHECKPOINTS_DIR"] = str(BASE_DIR / "model")
os.environ["JAEHYUN_DF_ARENA_DIR"] = str(BASE_DIR / "model" / "df_arena_1b")
os.environ["JAEHYUN_LOFCZ_DIR"] = str(BASE_DIR / "model" / "lofcz_ai_music_detector")

sys.path.insert(0, str(BASE_DIR))
from src import config  # noqa: E402
from src.pipeline import AudioPresencePipeline  # noqa: E402

DATA_DIR = Path(os.environ.get("DACON_DATA_DIR", str(BASE_DIR / "data")))
TEST_DIR = DATA_DIR / "test"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
OUTPUT_DIR = Path(os.environ.get("DACON_OUTPUT_DIR", str(BASE_DIR / "output")))
OUTPUT_PATH = OUTPUT_DIR / "submission.csv"

# Separated-route voice head (2026-09-29, TODO #10): on files routing
# separated, the DF-Arena voice-fake score comes from a Conformer fine-tuned on
# same-source real/fake pairs (last block + fc5, model/df_arena_side_conformer.pt),
# scored on the same Conformer input as the published model, and it feeds BOTH
# VOICE_FAKE_PROB and FILE_FAKE_PROB. The original route keeps the published
# model bit-for-bit. (probe_voice_paired.zip changed only the VOICE column:
# DACON VoiceEER 21.78 -> 21.56%.)
os.environ["JAEHYUN_DFA_SIDE_CONFORMER"] = str(BASE_DIR / "model" / "df_arena_side_conformer.pt")

PREDICTION_COLUMNS = ["FILE_FAKE_PROB", "VOICE_FAKE_PROB", "MUSIC_FAKE_PROB", "VOICE_PRESENT_PROB", "MUSIC_PRESENT_PROB"]
SUPPORTED_AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}

VOICE_FAKE_FALLBACK = 0.5  # used only when the voice detector is disabled entirely (enable_voice_fake_detection=False); routing no longer returns None (see music_routing.py CHANGELOG)
NO_MUSIC_FALLBACK = 0.5    # used only when ArtifactNet is disabled entirely (enable_artifactnet=False); routing no longer returns None (see music_routing.py CHANGELOG)

# If not None, round a *_PRESENT_PROB up to 1.0 for the FILE_FAKE_PROB
# combination only (never the reported *_PRESENT_PROB columns, which are
# separately AUC-scored for CPS) whenever it's already >= this threshold.
# Team finding (2026-09-15): dampening a confidently-present file's
# FILE_FAKE_PROB by a <1.0 presence score unnecessarily weakens the fake
# signal on files DACON's FileEER already scores as present -- a real
# submission with threshold=0.6 improved ADS. See SUBMISSION_LOG.md.
# Separate thresholds per component, because the two detectors are not
# equally trustworthy and rounding presence to 1.0 exposes a detector's raw
# score. DF-Arena (voice) measures EER 0.00% on the proxy set, so
# un-attenuating it is safe. lofcz (music) measures EER ~24% AND misfires on
# speech: run over 40 speech-only files it returned a median 0.680, higher
# than its median on genuine music (0.405) and close to its median on fake
# music (0.806) -- the harmonic structure of speech resembles the 1-8kHz
# regular peaks its "fakeprint" feature keys on. So a genuine speech-only
# file whose music presence merely crosses the music threshold gets that
# ~0.68 exposed directly as FILE_FAKE_PROB. Keeping the music threshold
# higher preserves the upside on confidently-music files while closing that
# MID-zone trap. See TODO.md #1.
PRESENCE_ROUND_THRESHOLD_VOICE = 0.2
PRESENCE_ROUND_THRESHOLD_MUSIC = 0.3


def round_presence_for_file_fake(present_prob, threshold):
    if threshold is None:
        return present_prob
    return 1.0 if present_prob >= threshold else present_prob


def find_audio_files(test_dir):
    files = sorted(
        (p for p in test_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS),
        key=lambda p: p.stem,
    )
    if not files:
        raise FileNotFoundError(f"No audio files found in {test_dir}")
    return files


def read_sample_submission(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)
    return columns, rows


def order_audio_files(audio_files, rows):
    by_id = {p.stem: p for p in audio_files}
    return [by_id[row["ID"]] for row in rows]


def main():
    t_start = time.time()
    print(f"Node data dir: {DATA_DIR}", flush=True)

    audio_files = find_audio_files(TEST_DIR)
    columns, rows = read_sample_submission(SAMPLE_SUBMISSION)
    audio_files = order_audio_files(audio_files, rows)
    print(f"Matched {len(audio_files)} test files", flush=True)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    cfg = config.PipelineConfig()
    cfg.enable_artifactnet = True
    cfg.music_fake_backend = "lofcz"  # see src/music_fake_lofcz.py + SUBMISSION_LOG.md
    cfg.lofcz.calibration_scale = 0.142962
    cfg.lofcz.calibration_bias = 3.097933
    cfg.force_artifactnet_original = False  # ablation: always score music-fake on the original mixture, never the separated instrumental
    cfg.force_voice_fake_original = False  # ablation: always score voice-fake on the original mixture, never the separated vocal
    cfg.enable_voice_fake_detection = True
    pipeline = AudioPresencePipeline(cfg=cfg, device=device)
    print(f"Pipeline loaded in {time.time() - t_start:.1f}s", flush=True)

    # Write incrementally (not all at once at the end) and never let one
    # file's failure (bad codec, corrupt audio, unexpected content) lose
    # every row already computed -- a hard timeout/kill from the eval
    # harness still leaves a partial, valid CSV this way, and any single
    # exception no longer aborts the whole run.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_f = OUTPUT_PATH.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(out_f, fieldnames=columns)
    writer.writeheader()

    num_failed = 0
    for i, (row, path) in enumerate(zip(rows, audio_files)):
        t0 = time.time()
        try:
            if pipeline.voice_fake_detector is not None:
                pipeline.voice_fake_detector.last_side_probability = None  # no stale value from the previous file
            result = pipeline.analyze_file(path)
            separated_route = bool(result["diagnostics"]["separated"])
            voice_present_prob = result["voice_probability"]
            music_present_prob = result["music_probability"]
            voice_fake_prob = result["voice_fake_probability"]
            if voice_fake_prob is None:  # only when the voice detector is disabled entirely -- routing itself never returns None now
                voice_fake_prob = VOICE_FAKE_FALLBACK
            music_fake_prob = result["music_fake_probability"]
            if music_fake_prob is None:  # only when ArtifactNet is disabled entirely -- routing itself never returns None now
                music_fake_prob = NO_MUSIC_FALLBACK
            side = (pipeline.voice_fake_detector.last_side_probability
                    if separated_route and pipeline.voice_fake_detector is not None else None)
            if side is not None:
                voice_fake_prob = side
            voice_present_for_combo = round_presence_for_file_fake(voice_present_prob, PRESENCE_ROUND_THRESHOLD_VOICE)
            music_present_for_combo = round_presence_for_file_fake(music_present_prob, PRESENCE_ROUND_THRESHOLD_MUSIC)
            file_fake_prob = max(voice_present_for_combo * voice_fake_prob, music_present_for_combo * music_fake_prob)
            route = result["route"]
        except Exception as e:
            print(f"[{i+1}/{len(rows)}] {row['ID']}: FAILED ({type(e).__name__}: {e}); writing neutral fallback row", flush=True)
            num_failed += 1
            voice_present_prob = music_present_prob = 0.5
            voice_fake_prob = VOICE_FAKE_FALLBACK
            music_fake_prob = NO_MUSIC_FALLBACK
            file_fake_prob = 0.5
            route = "ERROR"
            separated_route = False

        row["VOICE_PRESENT_PROB"] = round(voice_present_prob, 10)
        row["MUSIC_PRESENT_PROB"] = round(music_present_prob, 10)
        row["VOICE_FAKE_PROB"] = round(voice_fake_prob, 10)
        row["MUSIC_FAKE_PROB"] = round(music_fake_prob, 10)
        row["FILE_FAKE_PROB"] = round(file_fake_prob, 10)

        writer.writerow(row)
        out_f.flush()

        print(
            f"[{i+1}/{len(rows)}] {row['ID']}: route={route} "
            f"voice={voice_present_prob:.3f} music={music_present_prob:.3f} "
            f"voice_fake={voice_fake_prob:.3f} music_fake={music_fake_prob:.3f} "
            f"file_fake={file_fake_prob:.3f} ({time.time()-t0:.2f}s)",
            flush=True,
        )

    out_f.close()
    if num_failed:
        print(f"WARNING: {num_failed}/{len(rows)} files failed and got neutral fallback rows", flush=True)

    print(f"Wrote {len(rows)} predictions to {OUTPUT_PATH} in {time.time()-t_start:.1f}s total", flush=True)


if __name__ == "__main__":
    main()
