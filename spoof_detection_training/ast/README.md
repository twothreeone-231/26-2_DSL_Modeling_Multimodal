# AST 학습 파이프라인

우리 멀티모달팀의 DeepVoice 모델 학습 설정 중 AST 쪽 코드입니다.

> **파이프라인 구조가 바뀌었습니다 (2026-09-22)**: AST는 이제 **presence(VP/MP)만**
> 담당합니다. VF/MF/FF fake 탐지는 [`yubeen/aasist/`](../aasist/)의 AASIST 모델이
> 맡습니다. 전체 그림은 [`yubeen/README.md`](../README.md) 참고. 이 아래 `model.py`/
> `train.py`(`DeepVoiceAST`, VF/MF/FF 3-head)는 AST 혼자 fake까지 예측하던 이전 버전으로,
> 지금은 제출 경로가 아니라 비교용 baseline으로만 남아 있습니다.

## 📁 구조

```
yubeen/ast/
├── model.py              # (레거시) DeepVoiceAST 3-head 분류기 (VF, MF, FF) -- baseline
├── train.py               # (레거시) 위 모델 학습 스크립트 -- baseline
├── presence.py            # ★ presence(VP/MP) 학습 + AASIST와 결합한 5-head 추론/제출
├── dataset.py             # 혼합 데이터셋 로더 (yusung/data_mixed_v5)
├── extract_embeddings.py # (선택) CPU 친화적: 임베딩 캐싱 후 헤드만 학습
├── train_head.py         # (선택) extract_embeddings.py와 함께 사용
├── slurm/
│   ├── train_ast.sh       # (레거시) train.py용 sbatch
│   ├── train_presence.sh  # ★ presence.py 학습용 sbatch
│   ├── eval_ast.sbatch    # (레거시) evaluation.py용 sbatch
│   └── logs/              # slurm 로그
└── checkpoints/           # 학습 체크포인트 (ast_best.pt는 레거시, presence_best.pt가 현재 경로)
```

## 🎯 학습 목표 (멀티라벨 분류)

각 10초 클립에서 독립적으로 판정:
- **VF** (Voice Fake): 음성 진위 → 진짜 음성 / 가짜 음성 (음성이 없는 클립은 label=-1, mask로 손실/평가에서 제외)
- **MF** (Music Fake): 음악 진위 → 진짜 음악 / 가짜 음악 (음악이 없는 클립은 label=-1, 동일하게 제외)
- **FF** (File Fake): 파일에 존재하는 성분 중 가짜가 하나라도 있음

v3 데이터셋은 음성/음악이 항상 함께 있지 않고 9개 라벨 그룹으로 구성됨
(`voice_only` / `voice_fake_only` / `music_only` / `music_fake_only` /
`voice_music` / `voice_music_fake` / `voice_fake_music` /
`voice_fake_music_fake` / `noise_only`). VF/MF의 -1 라벨은 대회 평가
산식이 "Voice EER은 음성이 존재하는 샘플에서만, Music EER은 음악이
존재하는 샘플에서만" 계산한다는 규칙과 그대로 대응된다.

체크포인트 선택 기준도 단순 정확도가 아니라 대회 산식과 동일하게
계산한 EER 기반 지표(`ads_like` = 0.5*(1-FileEER) + 0.2*(1-VoiceEER) +
0.3*(1-MusicEER), presence는 `cps_like` = 평균 ROC-AUC)를 사용한다.

## 🚀 사용법

### 1. 환경 진단 (GPU 동작 확인)

```bash
sbatch slurm/check_env.sh
# 결과 확인: tail -f slurm/logs/ast-envcheck_*.out
```

### 2. 학습 실행

아래 제출 스크립트는 명령을 즉시 반환합니다. GPU가 아직 없으면 Slurm 대기열에서
기다리고, GPU가 할당되는 순간 `train.py`를 자동으로 실행하므로 터미널을 닫고
기다려도 됩니다.

#### 기본 (Backbone 고정)
```bash
slurm/submit_train.sh
# → checkpoints/job_JOBID/training_report.json
```

#### Backbone 파인튜닝 (epoch 3부터)
```bash
slurm/submit_train.sh --unfreeze-epoch 3
```

#### 빠른 테스트 (2000 샘플, 2 epoch)
```bash
slurm/submit_train.sh --limit 2000 --epochs 2
```

직접 제출해야 하는 경우에도 기존 명령인 `sbatch slurm/train_ast.sh`를 사용할 수
있습니다. 제출 후에는 출력된 job ID로 상태와 로그를 확인합니다.

### 3. 모니터링

```bash
# 제출된 job 확인
squeue -u $USER

# 로그 실시간 보기
tail -f slurm/logs/ast-train_*.out

# 결과 확인
cat checkpoints/job_JOBID/training_report.json | jq '.test'
```

## 📊 Slurm 리소스

- **Partition**: `partition1` (hpc-stat1)
- **GPU**: 1 × V100 (16GB VRAM)
- **CPU**: 4 cores, 32GB RAM
- **Time**: 24:00:00 (충분)

현재 partition1이 다른 팀의 오래된 job들로 혼잡 상태입니다.
우선순위가 낮으면 대기 시간이 길 수 있습니다.

## 🔧 train.py 옵션

```
--epochs INT              기본값: 10
--batch-size INT          기본값: 16
--num-workers INT         기본값: 4 (DataLoader 병렬화)
--head-lr FLOAT           기본값: 1e-3 (분류기 학습률)
--backbone-lr FLOAT       기본값: 1e-5 (backbone 학습률, unfrozen 시만)
--weight-decay FLOAT      기본값: 1e-4
--dropout FLOAT           기본값: 0.2
--unfreeze-epoch INT      지정하면 그 epoch부터 backbone 학습 시작
--amp                     혼합 정밀도 (bf16) 활성화
--eval-test               test split 평가
--limit INT               train/val 샘플 수 제한 (빠른 테스트용)
--out DIR                 체크포인트 저장 경로
```

## 📈 Expected Results

로컬 val/test EER·AUC 수치는 v3로 재학습 후 다시 채울 것 (이전 v2 4-combo
기준 combo accuracy 0.85~0.90은 실제 대회 리더보드 ADS와 상관관계가 없었음 —
음성만/음악만/노이즈만 있는 케이스를 학습 데이터가 전혀 포함하지 않았던 것이
원인. v3는 9개 라벨 그룹으로 이 gap을 메움).

## 🔗 데이터 경로

- **혼합 데이터 (v5, 최신)**: `/cluster/workspace/deepvoice/yusung/data_mixed_v5/` (라벨 스키마는 v3와 동일 -- 자세한 버전 이력은 `yusung/DATASET_GUIDE_V1_TO_V5.md` 참고)
  - Audio: `audio/{train,val,test}/{mixed_id}.wav`
  - Metadata: `metadata/mixed_{train,val,test}.csv`
    (컬럼: `mixed_id, split, VP, MP, VF, MF, FF, voice_source, music_source, voice_dataset, music_dataset`)

## 🔗 AASIST와 결합 (5-head 추론 / 제출 CSV)

`presence.py`가 fake 쪽은 [`yubeen/aasist/`](../aasist/)에서 학습한 `aasist_best.pt`를,
presence 쪽은 이 디렉토리의 `presence_best.pt`를 같이 로드합니다:

```bash
python presence.py \
  --audio-dir /path/to/test_audio \
  --fake-checkpoint ../aasist/checkpoints/job_JOBID/aasist_best.pt \
  --presence-checkpoint checkpoints/job_JOBID/presence_best.pt \
  --submission-out output/submission.csv
```

`--fake-checkpoint`/`--presence-checkpoint`를 생략하면 `checkpoints/`,
`yubeen/ast/checkpoints/`, `yubeen/aasist/checkpoints/`(파일명 `aasist_best.pt`/
`presence_best.pt`)를 순서대로 자동 탐색합니다.

## 💡 알려진 제약

1. **GPU 부족**: partition1이 다른 팀에 의해 많이 사용 중입니다.
   - 현재 (02:00 KST) 실행 중 job: 14개 (CPU 156코어 점유)
   - 새 job의 우선순위가 낮아 대기 시간이 길 수 있습니다.

2. **CPU 할당 경쟁**: 
   - sbatch 요청 cpu-per-task를 너무 크게 하지 마세요.
   - num-workers를 cpu-per-task보다 작게 유지하세요.

3. **캐시**: HF_HOME은 `/cluster/workspace/.cache/huggingface`입니다 (공유).

---

작성: 2026-09-17 02:00 KST
