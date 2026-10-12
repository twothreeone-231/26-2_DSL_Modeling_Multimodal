"""AudioSet-527 ontology loading and Voice / Music label-group definitions.

Design constraints (see AUDIO_PRESENCE_SEPARATION_TASK.md section 5):
  * Class indices are never hardcoded. We load the official
    `class_labels_indices.csv` (index, mid, display_name) and resolve label
    *names* -> indices at runtime.
  * We never use a single class ("Speech" alone, "Music" alone) as the whole
    voice/music signal. VOICE_LABELS and MUSIC_LABELS are curated groups of
    AudioSet display names.
  * Every requested label name is validated against the loaded ontology at
    import time; an unknown name raises immediately instead of silently
    being ignored.
  * VOICE_LABELS and MUSIC_LABELS are disjoint: vocal/singing classes are
    voice evidence, never music evidence.
"""
from __future__ import annotations

import csv
import logging
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from . import config

logger = logging.getLogger(__name__)

NUM_AUDIOSET_CLASSES = 527


@dataclass(frozen=True)
class AudioSetOntology:
    index_to_mid: Tuple[str, ...]
    index_to_name: Tuple[str, ...]
    name_to_index: Dict[str, int]
    mid_to_index: Dict[str, int]

    def __len__(self) -> int:
        return len(self.index_to_name)


def _download_ontology(dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading official AudioSet ontology from %s", config.AUDIOSET_ONTOLOGY_URL)
        with urllib.request.urlopen(config.AUDIOSET_ONTOLOGY_URL, timeout=15) as resp:
            data = resp.read()
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)
        return True
    except Exception as exc:  # noqa: BLE001 - network can fail for many reasons
        logger.warning("Could not download AudioSet ontology (%s); will try fallback copy.", exc)
        return False


def _resolve_ontology_csv_path() -> Path:
    cache_path = config.AUDIOSET_ONTOLOGY_CACHE
    if cache_path.is_file():
        return cache_path

    if _download_ontology(cache_path):
        return cache_path

    fallback = config.AUDIOSET_ONTOLOGY_FALLBACK
    if fallback.is_file():
        logger.info("Using vendored AudioSet ontology fallback at %s", fallback)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fallback, cache_path)
        return cache_path

    raise RuntimeError(
        "Could not obtain the official AudioSet class_labels_indices.csv "
        f"(no network, and no fallback found at {fallback})."
    )


_ONTOLOGY_CACHE: AudioSetOntology | None = None


def load_ontology(force_reload: bool = False) -> AudioSetOntology:
    global _ONTOLOGY_CACHE
    if _ONTOLOGY_CACHE is not None and not force_reload:
        return _ONTOLOGY_CACHE

    csv_path = _resolve_ontology_csv_path()
    rows: List[Tuple[int, str, str]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["index"]), row["mid"], row["display_name"]))
    rows.sort(key=lambda r: r[0])

    if len(rows) != NUM_AUDIOSET_CLASSES:
        raise ValueError(
            f"Expected {NUM_AUDIOSET_CLASSES} AudioSet classes, got {len(rows)} from {csv_path}"
        )
    expected_indices = list(range(NUM_AUDIOSET_CLASSES))
    actual_indices = [r[0] for r in rows]
    if actual_indices != expected_indices:
        raise ValueError(f"AudioSet ontology indices are not a contiguous 0..{NUM_AUDIOSET_CLASSES - 1} range")

    index_to_mid = tuple(r[1] for r in rows)
    index_to_name = tuple(r[2] for r in rows)
    name_to_index = {name: idx for idx, name in enumerate(index_to_name)}
    mid_to_index = {mid: idx for idx, mid in enumerate(index_to_mid)}

    _ONTOLOGY_CACHE = AudioSetOntology(
        index_to_mid=index_to_mid,
        index_to_name=index_to_name,
        name_to_index=name_to_index,
        mid_to_index=mid_to_index,
    )
    logger.info("Loaded AudioSet ontology (%d classes) from %s", len(index_to_name), csv_path)
    return _ONTOLOGY_CACHE


# ---------------------------------------------------------------------------
# Voice / Music label groups.
#
# Curated from the official AudioSet ontology display names. Grouping mirrors
# the project's Voice/Music definition (see task doc section 1): sung vocals
# are Voice, never Music; instruments/genres/production concepts are Music.
# Every name below is validated against the ontology in `validate_labels()`.
# ---------------------------------------------------------------------------

VOICE_LABELS: Tuple[str, ...] = (
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Babbling",
    "Speech synthesizer",
    "Singing",
    "Choir",
    "Male singing",
    "Female singing",
    "Child singing",
    "Synthetic singing",
    "Rapping",
    "Humming",
    "Chant",
    "Mantra",
    "Yodeling",
    "Whispering",
)

MUSIC_LABELS: Tuple[str, ...] = (
    "Music",
    "Musical instrument",
    "Plucked string instrument",
    "Guitar",
    "Electric guitar",
    "Bass guitar",
    "Acoustic guitar",
    "Steel guitar, slide guitar",
    "Tapping (guitar technique)",
    "Strum",
    "Banjo",
    "Sitar",
    "Mandolin",
    "Zither",
    "Ukulele",
    "Keyboard (musical)",
    "Piano",
    "Electric piano",
    "Organ",
    "Electronic organ",
    "Hammond organ",
    "Synthesizer",
    "Sampler",
    "Harpsichord",
    "Percussion",
    "Drum kit",
    "Drum machine",
    "Drum",
    "Snare drum",
    "Rimshot",
    "Drum roll",
    "Bass drum",
    "Timpani",
    "Tabla",
    "Cymbal",
    "Hi-hat",
    "Wood block",
    "Tambourine",
    "Rattle (instrument)",
    "Maraca",
    "Gong",
    "Tubular bells",
    "Mallet percussion",
    "Marimba, xylophone",
    "Glockenspiel",
    "Vibraphone",
    "Steelpan",
    "Orchestra",
    "Brass instrument",
    "French horn",
    "Trumpet",
    "Trombone",
    "Bowed string instrument",
    "String section",
    "Violin, fiddle",
    "Pizzicato",
    "Cello",
    "Double bass",
    "Wind instrument, woodwind instrument",
    "Flute",
    "Saxophone",
    "Clarinet",
    "Harp",
    "Bell",
    "Accordion",
    "Bagpipes",
    "Didgeridoo",
    "Shofar",
    "Theremin",
    "Background music",
    "Theme music",
    "Jingle (music)",
    "Soundtrack music",
    "Video game music",
    "Pop music",
    "Hip hop music",
    "Rock music",
    "Heavy metal",
    "Punk rock",
    "Grunge",
    "Progressive rock",
    "Rock and roll",
    "Psychedelic rock",
    "Rhythm and blues",
    "Soul music",
    "Reggae",
    "Country",
    "Swing music",
    "Bluegrass",
    "Funk",
    "Folk music",
    "Middle Eastern music",
    "Jazz",
    "Disco",
    "Classical music",
    "Opera",
    "Electronic music",
    "House music",
    "Techno",
    "Dubstep",
    "Drum and bass",
    "Electronica",
    "Electronic dance music",
    "Ambient music",
    "Trance music",
    "Music of Latin America",
    "Salsa music",
    "Flamenco",
    "Blues",
    "New-age music",
    "Music of Africa",
    "Music of Asia",
    "Carnatic music",
    "Music of Bollywood",
    "Traditional music",
    "Independent music",
    "Dance music",
)


def validate_labels(names: Sequence[str], ontology: AudioSetOntology, group_name: str) -> None:
    missing = [name for name in names if name not in ontology.name_to_index]
    if missing:
        raise ValueError(
            f"{group_name}: the following labels do not exist in the AudioSet ontology "
            f"({len(ontology)} classes): {missing}"
        )


def validate_label_groups(
    voice_labels: Sequence[str] = VOICE_LABELS,
    music_labels: Sequence[str] = MUSIC_LABELS,
    ontology: AudioSetOntology | None = None,
) -> AudioSetOntology:
    """Validate both groups exist in the ontology and are disjoint. Returns the ontology."""
    if ontology is None:
        ontology = load_ontology()
    validate_labels(voice_labels, ontology, "VOICE_LABELS")
    validate_labels(music_labels, ontology, "MUSIC_LABELS")
    overlap = set(voice_labels) & set(music_labels)
    if overlap:
        raise ValueError(f"VOICE_LABELS and MUSIC_LABELS must be disjoint, overlap found: {overlap}")
    return ontology


def label_indices(names: Sequence[str], ontology: AudioSetOntology | None = None) -> List[int]:
    if ontology is None:
        ontology = load_ontology()
    validate_labels(names, ontology, "label group")
    return [ontology.name_to_index[name] for name in names]
