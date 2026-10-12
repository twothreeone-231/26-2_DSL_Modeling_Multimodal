# AST + AASIST

AST presence와 AASIST fake-detection 학습 파이프라인 모음입니다.

- `ast/`: VP/MP presence 모델
- `aasist/`: VF/MF/FF fake 탐지 모델
- `common.py`: 두 학습 파이프라인의 공통 코드

학습 제출:

```bash
sbatch spoof_detection_training/ast/slurm/train_presence.sh
sbatch spoof_detection_training/aasist/slurm/train_aasist.sh
```

> 학습 체크포인트(`checkpoints/`), 변환된 제출용 가중치(`ast/submit/model/*.pt`,
> `*.safetensors`), 학습 로그는 용량 때문에 이 레포에 포함하지 않았습니다.
