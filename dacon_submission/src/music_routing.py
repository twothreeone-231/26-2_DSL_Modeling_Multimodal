"""Unified voice/music presence-confidence routing: decides, from the
*original* (pre-separation) AudioMosaic voice/music probabilities, (a)
whether BS PolarFormer separation runs at all, and (b) what input each
fake-detector (ArtifactNet for music, WavLM for voice) should use --
the separated stem when separation ran, otherwise the original mixture.

Team-designed scheme (see jaehyun/README.md routing section for the full
derivation and musan-based empirical validation of the thresholds):

    zone(P) = LOW   if P < low_threshold        (confidently absent)
              HIGH  if P >= high_threshold       (confidently present)
              MID   otherwise                    (uncertain)

    both zones LOW:
        neither present -> no separation; both detectors still run on the
        original mixture (see "never skip a detector" below)

    voice_zone == LOW (music_zone MID or HIGH):
        no vocal interference risk -> ArtifactNet(original) directly, no
        separation needed; WavLM ALSO runs on the original (see below)

    music_zone == LOW (voice_zone MID or HIGH):
        no musical interference risk -> WavLM(original) directly, no
        separation needed; ArtifactNet ALSO runs on the original (see below)

    otherwise (neither zone is LOW, i.e. both >= low_threshold):
        can't trust either component alone -> BS PolarFormer separation
        runs; ArtifactNet(separated instrumental) and WavLM(separated
        vocal) both run.

CHANGELOG (2026-09-14, tried and reverted): briefly widened separation to
also run whenever voice_zone alone was not LOW (even with music_zone ==
LOW), reasoning from three real-leaderboard ablations that showed forcing
the voice detector to always use the original mixture cost -2.02pp ADS
(vs only -0.55pp for the music detector) -- see SUBMISSION_LOG.md
2026-09-14 rows for that ablation set. A real submission of the widened
version (SUBMISSION_LOG.md, same date, "widened_voice_routing" row)
measurably REGRESSED instead: -0.50pp ADS vs the then-best submission,
despite CPS improving slightly (+0.08pp) -- the first submission in this
whole log where CPS actually moved, confirming the routing change itself
(not just the detector swap) was responsible. Root-cause read: the
original ablations measured separation's benefit only on files that
already triggered the old BOTH condition (voice AND music both genuinely
present) -- a population where there is real musical interference to
remove. Widening added separation to a *different* population (voice
present, music confidently ABSENT), where there is nothing to separate
out; BS PolarFormer's own artifacts (residual noise, phase distortion)
then only ever hurt DF-Arena's input with no offsetting information gain.
Conclusion: separation helps the voice detector only when music is
ALSO genuinely present, not from voice presence alone -- reverted to the
both-not-LOW condition below.

CHANGELOG (real DACON regression, fixed): an earlier version of this
scheme returned None (-> a neutral 0.5 fallback at submission time) for
a component whenever its own zone was LOW, on the theory that a
"confidently absent" call needn't bother running that detector. This
measurably hurt the real leaderboard score (submission scored 2026-09-02:
ADS dropped from 0.643 to 0.571 vs. the pre-WavLM submission, while CPS
was essentially unchanged at ~0.998 -- confirming the regression was in
fake-detection, not presence detection). The reasoning was backwards:
DACON's Voice/Music EER is scored on every ground-truth-*present* sample
regardless of what zone we predicted, so a wrong "confidently absent" call
still needs the detector's real signal on that (rare) miss -- a
guaranteed-neutral 0.5 is worse than a real (even imperfect, off-domain)
score whenever we might be wrong, and costs nothing when we're right
(that sample isn't scored on that metric at all, per the official rule
"Voice EER은 음성이 존재하는 샘플에서만... 계산됩니다"). Both detectors are
cheap enough (WavLM ~0.1s/file, ArtifactNet sub-second on GPU) that always
running them is not the bottleneck BS PolarFormer separation was -- so
`artifactnet_input`/`voice_detector_input` are no longer Optional; a
detector is applied to *something* (original or separated) on every file.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import config

Zone = Literal["LOW", "MID", "HIGH"]
DetectorInput = Literal["original", "separated_instrumental", "separated_vocal"]


def classify_zone(probability: float, cfg: config.RoutingConfig) -> Zone:
    if probability < cfg.low_threshold:
        return "LOW"
    if probability >= cfg.high_threshold:
        return "HIGH"
    return "MID"


@dataclass
class ProcessingPlan:
    music_zone: Zone
    voice_zone: Zone
    run_separation: bool
    artifactnet_input: DetectorInput
    voice_detector_input: DetectorInput

    @property
    def label(self) -> str:
        """Human-readable summary for logging/CLI display (not used for
        control flow -- the zones/inputs above are the source of truth)."""
        if self.voice_zone == "LOW" and self.music_zone == "LOW":
            return "NEITHER"
        if self.voice_zone == "LOW":
            return "MUSIC_ONLY"
        if self.music_zone == "LOW":
            return "VOICE_ONLY"
        return "BOTH"


def decide_processing_plan(
    voice_probability: float,
    music_probability: float,
    routing_cfg: config.RoutingConfig,
) -> ProcessingPlan:
    music_zone = classify_zone(music_probability, routing_cfg)
    voice_zone = classify_zone(voice_probability, routing_cfg)

    if voice_zone != "LOW" and music_zone != "LOW":
        # Neither component is confidently absent -> can't trust either
        # alone; separate and score both on their own isolated stem.
        return ProcessingPlan(
            music_zone, voice_zone, run_separation=True,
            artifactnet_input="separated_instrumental", voice_detector_input="separated_vocal",
        )

    # At least one component is confidently absent -> no interference risk
    # either way, so separation buys nothing here (see 2026-09-14 CHANGELOG
    # above: widening separation to voice-present-but-music-absent files
    # was tried on the real leaderboard and regressed). Both detectors
    # still run on the original mixture -- never skipped -- per the
    # CHANGELOG below.
    return ProcessingPlan(
        music_zone, voice_zone, run_separation=False,
        artifactnet_input="original", voice_detector_input="original",
    )
