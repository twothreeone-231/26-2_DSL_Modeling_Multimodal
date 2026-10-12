# 26-2_DSL_Modeling_Multimodal

> 임시 안내용 README입니다. 프로젝트 설명 README는 별도로 다시 작성할 예정이며,
> 지금은 1차 결과물 검토를 위한 폴더 구조 안내만 담고 있습니다.

## 폴더 구조

- `web_demo/` — 오디오를 업로드하면 AI 생성 여부를 분석해주는 FastAPI 웹 데모
  (`backend/` + `frontend/`). 탐지 파이프라인은 `pipeline_src/src/`에 번들.
- `dacon_submission/` — 대회 제출용 배치 추론 엔트리포인트(`script.py`). 탐지
  파이프라인은 `src/`에 번들 (`web_demo/pipeline_src/src/`와 동일 코드).
- `data_preprocessing/` — 학습 데이터 전처리 파이프라인 v1~v5 (버전별 핵심 변경사항
  1개 파일씩).
- `spoof_detection_training/` — AST(presence) / AASIST(fake 탐지) 학습 코드.

## 제외한 것

모델 가중치(`.pt`, `.bin`, `.safetensors`, `.onnx` 등, 수백MB~4.6GB), 학습
체크포인트 폴더, Hugging Face 캐시, 실행 로그는 레포 용량 때문에 포함하지
않았습니다. 각 모듈 안의 `README.md`에 원본 경로나 구성이 설명되어 있습니다.
