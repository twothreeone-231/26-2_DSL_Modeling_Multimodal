#!/usr/bin/env bash
# ===========================================================================
# [목적] SLURM 계산 노드에서 딥보이스 탐지 웹 데모 서버를 띄운다.
#
# [사용법] 계산 노드(srun으로 확보한 셸) 안에서:
#     bash run_on_slurm.sh
#
# 노트북에서는 아래로 터널을 뚫고 http://localhost:18342 접속:
#     ssh -N -L 18342:<계산노드IP>:18342 dsl07@<로그인노드>
#   (계산노드 IP는  getent hosts $(hostname) | awk '{print $1}'  로 확인)
#
# [왜 마이크가 localhost에서만 되는가] 브라우저는 보안상 HTTPS 또는 localhost
# 에서만 마이크 접근을 허용한다. 터널로 localhost에 접속하면 인증서 없이도
# 녹음이 동작한다. 계산 노드 IP로 직접 접속하면 녹음 기능이 막힌다.
# ===========================================================================
set -euo pipefail

# ── 1. 파이프라인 코드 경로 ────────────────────────────────────────────────
# [고정 사본을 쓰는 이유] jaehyun 라이브 dev 폴더를 직접 가리키면, 발표 직전에
# 그쪽 코드가 수정되는 순간 데모가 깨진다. 데모용으로 복사해 고정해두고 쓴다.
# 코드는 100KB 남짓이라 복사 비용이 없고, 무거운 가중치는 아래 env var로 공유한다.
#
# [/mnt/data1 인 이유] 로그인 노드(hpcmaster)는 이 공유 스토리지를 /data1/...로
# 보지만, 이 스크립트가 실제로 실행되는 계산 노드(hpc-stat1 등)는 /data1/...가
# 아예 존재하지 않고 같은 스토리지가 /mnt/data1/...로만 마운트되어 있다
# (실측 확인됨: /data1로 두면 "경로 없음"으로 즉시 중단됨).
PIPELINE_ROOT="${PIPELINE_ROOT:-/mnt/data1/dsl07/deepvoice/sewon/web_demo/pipeline_src}"

# ── 2. 모델 가중치 경로 ────────────────────────────────────────────────────
# src/config.py가 이 환경변수들로 경로를 받도록 이미 설계돼 있어서,
# 코드와 가중치가 서로 다른 디렉터리에 있어도 수정 없이 연결된다.
DV_BASE="/mnt/data1/dsl07/deepvoice"
export JAEHYUN_CHECKPOINTS_DIR="${DV_BASE}/checkpoints"
export JAEHYUN_HF_CACHE_DIR="${DV_BASE}/checkpoints/hf_cache"
export JAEHYUN_DF_ARENA_DIR="${DV_BASE}/model/df_arena_1b"
export JAEHYUN_LOFCZ_DIR="${DV_BASE}/model/lofcz_ai_music_detector"
# component-isolation probe (jaehyun outputs/probe_voice_paired.zip, 2026-09-27):
# fine-tuned Conformer (last block + fc5), scored alongside the published
# DF-Arena model on separated-vocal input. src/voice_fake_detector.py loads
# it only if this file exists; harmless to leave set otherwise.
export JAEHYUN_DFA_SIDE_CONFORMER="${DV_BASE}/model/df_arena_side_conformer.pt"
export JAEHYUN_ONTOLOGY_CSV="${DV_BASE}/checkpoints/audioset_ontology/class_labels_indices.csv"

# 평가 서버와 동일하게 오프라인 모드로 둔다. 캐시된 가중치만 쓰므로 네트워크
# 상태(인증서 문제 등)에 영향을 받지 않아 발표 중 사고 위험이 줄어든다.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ── 3. 서버 설정 ───────────────────────────────────────────────────────────
export ANALYZER_MODE=model
export PIPELINE_ROOT
export PIPELINE_DEVICE="${PIPELINE_DEVICE:-cuda}"
PORT="${PORT:-18342}"

# ── 4. 사전 점검 ───────────────────────────────────────────────────────────
# 27초짜리 모델 로딩을 시작하기 전에 경로부터 확인한다.
# 로딩 도중에 파일이 없어서 실패하면 원인을 찾기 번거롭기 때문.
fail=0
check() { [ -e "$2" ] || { echo "  [없음] $1: $2"; fail=1; }; }
echo "경로 점검:"
check "파이프라인 코드"   "${PIPELINE_ROOT}/src/pipeline.py"
check "DF-Arena 가중치"   "${JAEHYUN_DF_ARENA_DIR}/pytorch_model.bin"
check "lofcz 가중치"      "${JAEHYUN_LOFCZ_DIR}/model.safetensors"
check "HF 캐시"           "${JAEHYUN_HF_CACHE_DIR}/hub"
check "AudioSet 온톨로지" "${JAEHYUN_ONTOLOGY_CSV}"
[ "$fail" -eq 0 ] || { echo "경로 문제로 중단합니다."; exit 1; }
echo "  모두 확인됨"

# [IP 확인 방법 주의] getent hosts "$(hostname)"은 이 노드 자체의 /etc/hosts에
# 박힌 127.0.1.1류 루프백 항목을 집어올 수 있어(실측 확인됨) 터널 대상으로 못 쓴다.
# hostname -I는 실제 네트워크 인터페이스의 IP를 순서대로 내놓으므로 그 중 첫 값을 쓴다.
NODE_IP="$(hostname -I | awk '{print $1}')"
echo
echo "계산 노드 IP: ${NODE_IP}"
echo "노트북에서:  ssh -N -L ${PORT}:${NODE_IP}:${PORT} dsl07@<로그인노드>"
echo "그 다음:     http://localhost:${PORT}"
echo

# ── 5. 서버 실행 ───────────────────────────────────────────────────────────
# --host 0.0.0.0 : 기본값(127.0.0.1)은 그 노드 안에서만 접속을 받으므로,
#                  터널로 들어오는 연결을 받으려면 외부 인터페이스에도 열어야 한다.
# --workers 1    : 모델이 GPU 메모리 5GB를 점유한다. 워커를 늘리면 프로세스마다
#                  모델을 따로 로드해 메모리가 배로 든다. 데모에는 1개면 충분하다.
#
# [python -m uvicorn 인 이유] pip이 설치한 uvicorn 실행 스크립트의 shebang이
# 설치 당시 경로(/data1/...)로 고정 박혀있어, 계산 노드(/data1 없음)에서 그
# 스크립트를 직접 실행하면 "bad interpreter" 에러가 난다. 파이썬 인터프리터를
# 명시적으로 지정해 -m으로 실행하면 shebang을 안 타서 이 문제를 피한다.
cd "$(dirname "$0")"
exec "$(command -v python)" -m uvicorn main:app --host 0.0.0.0 --port "${PORT}" --workers 1
