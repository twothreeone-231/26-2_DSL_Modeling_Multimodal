# AASIST 학습 파이프라인

v5 혼합 데이터셋에서 **fake 탐지(VF/MF/FF)** 를 담당하는 AASIST 학습 코드. presence(VP/MP)는
AST 쪽(`yubeen/ast/`)이 맡습니다 — 전체 구조는 [`yubeen/README.md`](../README.md) 참고.

## 📁 구조

```
yubeen/aasist/
├── model.py              # 원본 AASIST 인코더 + 3-head 멀티라벨 classifier (VF/MF/FF)
├── dataset.py             # 혼합 데이터셋 로더 (raw waveform, 64600 샘플 고정 길이)
├── train.py               # GPU 학습 스크립트 (frozen/unfrozen backbone)
├── evaluation.py          # 저장된 checkpoint를 val/test에서 재평가
├── slurm/
│   ├── train_aasist.sh    # 학습 job (sbatch용)
│   └── eval_aasist.sbatch # 평가 job
└── checkpoints/           # 학습 체크포인트 (aasist_best.pt)
```

`yubeen/ast/`와 구조가 거의 동일합니다 — masked BCE loss, EER, ads_like 집계, group-balanced
subset 샘플링 같은 학습/평가 로직은 `yubeen/common.py`에 공용으로 있고, 여기 `train.py`는
모델(AASIST)과 데이터셋(raw waveform)만 다릅니다.

## 원본 코드와의 관계

`deepvoice/aasist/`는 [clovaai/aasist](https://github.com/clovaai/aasist) 원본 그대로 클론된
것입니다 (논문 저자 코드, 학습 없이 그대로 둠). `model.py`가 여기서 그 저장소의
`models/AASIST.py`를 import해서 인코더(SincConv + residual encoder + 2-branch graph
attention, 160-dim 임베딩)를 그대로 재사용하고, 사전학습된 `models/weights/AASIST.pth`
(ASVspoof bonafide/spoof용)를 기본으로 로드합니다. 원래 있던 `nn.Linear(160, 2)`
bonafide/spoof head는 쓰지 않고, 우리 문제(VF/MF/FF 멀티라벨)에 맞는
`nn.Linear(160, 3)` head를 새로 얹었습니다.

입력은 원본 논문 코드의 `data_utils.py`에 있는 `pad`/`pad_random` 함수를 그대로 가져다
씁니다 — train은 4.04초(64600 샘플) 랜덤 crop(augmentation), val/test는 앞에서부터
고정 crop(재현 가능한 평가).

## 🎯 학습 목표

각 10초 클립에서 독립적으로 판정 (yubeen/ast/train.py와 동일한 정의):
- **VF** (Voice Fake): 음성이 없는 클립은 label=-1, mask로 손실/평가에서 제외
- **MF** (Music Fake): 음악이 없는 클립은 동일하게 제외
- **FF** (File Fake): 존재하는 성분 중 가짜가 하나라도 있음

체크포인트 선택 기준은 EER 기반 `ads_like` = 0.5*(1-FileEER) + 0.2*(1-VoiceEER) +
0.3*(1-MusicEER).

## 🚀 사용법

```bash
# 학습 (기본: frozen backbone, 10 epoch)
sbatch slurm/train_aasist.sh
# → checkpoints/job_JOBID/aasist_best.pt, training_report.json

# backbone 파인튜닝
sbatch slurm/train_aasist.sh --unfreeze-epoch 3

# 빠른 스모크테스트
sbatch slurm/train_aasist.sh --limit 200 --epochs 1
```

```bash
# 저장된 checkpoint 재평가
sbatch slurm/eval_aasist.sbatch --checkpoint checkpoints/job_JOBID/aasist_best.pt
```

## 🔧 train.py 옵션

`yubeen/ast/train.py`와 동일한 인자 세트: `--epochs`, `--batch-size`, `--num-workers`,
`--head-lr`, `--backbone-lr`, `--weight-decay`, `--dropout`, `--unfreeze-epoch`, `--amp`,
`--eval-test`, `--limit`, `--val-limit`, `--out`.

## 🔗 데이터 경로

- **혼합 데이터 (v5, 최신)**: `/data1/dsl07/deepvoice/yusung/data_mixed_v5/`
  (yubeen/ast와 완전히 동일한 데이터셋 — presence/fake가 같은 클립을 보고 다른 라벨을 예측.
  버전 이력은 `yusung/DATASET_GUIDE_V1_TO_V5.md` 참고)

## 결합 (5-head 추론)

학습이 끝나면 `yubeen/ast/presence.py`가 이 checkpoint(`aasist_best.pt`)와 AST의
`presence_best.pt`를 같이 로드해서 VF/MF/FF/VP/MP 5-head 결과와 대회 제출 CSV를 만듭니다:

```bash
cd ../ast
python presence.py \
  --audio-dir /path/to/test_audio \
  --fake-checkpoint ../aasist/checkpoints/job_JOBID/aasist_best.pt \
  --presence-checkpoint checkpoints/job_JOBID/presence_best.pt \
  --submission-out output/submission.csv
```

---

작성: 2026-09-22
