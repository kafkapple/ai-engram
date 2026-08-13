# 실행 환경 — 개인 윈도우 서버 (gpu03 대체)

> 2026-08-13 작성. gpu03 접속은 2026-08-17 오프보딩으로 종료.
> 이 문서 = AI Engram 실험 실행환경 SSOT. 구 `gpu03_env_bootstrap.sh`는 참조용으로만 보존.

## tldr

- gpu03에 남은 AI Engram 자산 **없음** — 코드·HF 캐시·체크포인트·로그 모두 이미 소거됨(260813 실측). 회수 대상 0건.
- 자산 정본 = GitHub 2개 repo + 볼트. 실행환경만 재구축하면 재개 가능.
- 개인 서버 = RTX 3060 12GB. **1B 배터리 전량 실행 가능 · 3B 재학습 계열은 VRAM 부족으로 불가.**

## 1. gpu03 최종 상태 (260813 실측)

| 항목 | 경로 | 상태 |
|---|---|---|
| 코드 | `~/dev/ai-engram` | 없음 (디렉토리 자체 부재) |
| 체크포인트 | `/node_data/joon/checkpoints/ai_engram` | 없음 (`FaceLift`만 잔존) |
| HF 캐시 | `/node_data_2/joon/hf_cache/hub` | 없음 (`FLUX.1-dev`만 잔존) |
| 실행 로그 | `/node_data/joon/logs_ai_engram` | 없음 |
| conda env | `~/anaconda3/envs/ai-engram` | 잔존 → 본 repo에 export 완료 |

- 회수 완료물 = `environment.gpu03.yml`, `pip-freeze.gpu03.txt` (결과 재현 시 버전 대조용).
- 핵심 핀: python 3.11.15 · torch 2.12.1 · transformers 5.13.0 · datasets 5.0.0 · numpy 2.4.6.

## 2. 자산 소재 (3중화 유지)

| 자산 | 위치 |
|---|---|
| 실험 스크립트 + `_repro_out/` 결과 13종 + HTML 리포트 | GitHub `kafkapple/ai-engram-repro` (private) |
| engram 패키지 (포크) | GitHub `kafkapple/ai-engram` (public) |
| 분석·마일스톤 노트 | 볼트 `30_Projects/_Hold/2607_AI_Engram_Audit/` |
| 로컬 미러 | mac `~/dev/ai-engram-repro`, `~/dev/ai-engram` (260813 복원) |

- 모델·데이터셋은 전부 HF Hub 공개 repo → 재다운로드로 복구. 로컬 보관 불요.

## 3. 개인 서버 실측 사양 (260813)

- 호스트: `DESKTOP-U91M55P` · SSH 별칭 `win`(PowerShell) / `wsl`(Ubuntu).
- GPU: RTX 3060 **12288 MiB**, driver 591.86 (`ssh win nvidia-smi`).
- WSL2: Ubuntu, RAM 15GB, `/` 여유 687G.
- 실행 위치 = **WSL** (스크립트가 bash·POSIX 경로 전제).

## 4. 부트스트랩

```bash
ssh wsl
cd ~/dev/ai-engram-repro && bash win_env_bootstrap.sh      # conda env + torch + 패키지 + T0 테스트
conda activate ai-engram && python win_prefetch_assets.py  # 1B 3종 + TOFU 4 config (~8GB)
#                          python win_prefetch_assets.py --with-3b   # 3B 추론용 추가시
```

### 검증 결과 (260814 실측, `_boot2.log`)

- `engram 0.8.0 | torch 2.12.1+cu130 | cuda True | dev NVIDIA GeForce RTX 3060`
- `vram free/total: 11.0/12.0 GiB` · `bf16 supported: True`
- T0 CPU 유닛테스트 **52 passed, 1 skipped** (24.1s)

### 버전 드리프트 (gpu03 대비)

| 패키지 | gpu03 (결과 생성 시점) | 개인 서버 (260814) |
|---|---|---|
| python / torch / numpy | 3.11.15 / 2.12.1 / 2.4.6 | 동일 |
| transformers | 5.13.0 | **5.15.0** |
| datasets | 5.0.0 | **5.0.1** |

- 기존 `_repro_out/` 수치와 직접 비교하는 재실행에는 핀 권장: `pip install "transformers==5.13.0" "datasets==5.0.0"`.
- 신규 실험(P2'·P4')은 드리프트 무관 — 핀 불요.

환경변수 (기본값은 스크립트 내장, 필요시만 override):

- `REPO_ROOT=~/dev/ai-engram` · `REPRO_ROOT=~/dev/ai-engram-repro`
- `HF_HOME=~/.cache/huggingface` · `CHECKPOINT_ROOT=~/checkpoints`
- gpu03 시절 `/node_data*` 경로는 스크립트에 **기본값으로만** 남아 있음 (전부 env override 가능).

## 5. VRAM 실행 가능성 (12GB 기준)

**병목은 가중치가 아니라 공분산 행렬이다.** `CovarianceCollector`가 대상 Linear마다 `float32` D×D를 잡고(`collectors.py:181`), 편집 시 forget·total 두 벌이 동시에 살아 있다. `target_modules=None`(기본)은 모든 Linear가 대상.

| 경로 | 지배 항목 | 1B (hidden 2048 / ffn 8192 / 16층) | 3B (hidden 3072 / ffn 8192 / 28층) |
|---|---|---|---|
| 편집(통계 수집→apply) | 공분산 2벌 | **12.1 GiB 실측** (260814 T2a, 12.3 중) | 🟡 ~29 GiB 추정 |
| full-param AdamW 재학습 | 옵티마이저 상태 | ~9.2 GiB (가중치+grad+state 2개) | ~23.9 GiB |

- 3B 추정 근거 = 구조비 스케일링. 층당 Σd² 는 1B `6×2048² + 8192²`, 3B `6×3072² + 8192²` → 전체 **2.35배**. 실측 앵커(1B 12.1 GiB)에 적용.
- **`storage_device="cpu"` 가 정식 완화책** — `EditorConfig`가 "covariances don't fit in VRAM (large/wide models)" 용도로 문서화한 옵션. 공분산을 시스템 RAM으로 내리면 3B **편집**은 가능해지나, WSL 기본 RAM 15GB로는 부족 → `.wslconfig`에 `memory=32GB` 필요.
- 재학습 경로는 `storage_device`와 무관 — 옵티마이저 상태 문제라 3B는 여전히 불가.
- 1B 완화책: `BS` 8→4 · `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` · 스크립트 내 `del`+`empty_cache()` 유지.

### 실행 시간 배율 (260814 관측)

- T2a NLL 프록시: gpu03 **~6분** → 3060 **87분+ 미완**(CPU 83분 소모, GPU 100%) = **15배 이상**.
- gpu03 기준 소요시간을 그대로 읽지 말 것. "~1일" 실험은 개인 서버에서 수일 규모.
- 원인 추정 🟡: 연산성능 격차 + VRAM 12.1/12.3 상시 점유로 인한 할당기 압박 + WSL 유효 1 vCPU.

### 재개 우선순위별 판정

| 항목 | 내용 | 개인 서버 |
|---|---|---|
| P1' | 3B 강도-매칭 C3L (`xval_3b`가 3B에 150-step full-param 파인튜닝) | ❌ 불가 |
| P2' | logit-lens 프로브 (추론 전용) | ✅ 가능 |
| P3' | R값 cluster-CI 재분석 (기존 JSON 재분석) | ✅ CPU로 가능 |
| P3' | frontier seed 증설 (1B 재학습) | ⚠️ 경계 |
| P4' | MNIST 클래스삭제 아날로그 | ✅ 가능 |

**3B 대안** (P1' 착수 시 택1):

1. 클라우드 GPU 시간제 대여 (A100 40GB, P1' 실행시간 ~1h) — 프로토콜 무변경, 권장.
2. LoRA 재학습으로 대체 — VRAM은 해결되나 "재학습 공격" 정의가 바뀌어 gpu03 결과와 **비교 불가**.
3. 1B 범위로 축소 — 모델 규모 교차 주장(dose non-transferability)을 포기.

- 8-bit optimizer·CPU offload는 해법 아님: 8-bit AdamW도 3B에서 ~19GB, CPU offload는 WSL RAM 15GB 초과.

## 6. 함정

- **conda-forge `python=3.11` env에는 pip이 없음** — `pip install`이 조용히 `/usr/bin/pip`으로 빠져 torch 전체가 `~/.local/lib/python3.10`에 설치됨(260813 실제 발생). 부트스트랩은 `pip`을 명시 생성하고 `$ENV_PREFIX/bin/python -m pip`으로만 호출.
- `conda activate`는 비대화형 셸에서 **exit 0을 반환하고도 PATH를 안 바꿀 수 있음** — 인터프리터는 절대경로로 바인딩할 것.
- WSL2는 `nvidia-smi`를 `/usr/lib/wsl/lib`에 두고 PATH에 넣지 않음 — 부트스트랩이 PATH·LD_LIBRARY_PATH에 주입.
- WSL에 `gh` 로그인 없음 → private repo(`ai-engram-repro`) push 전에 `gh auth login` 1회 필요. 최초 배포는 mac→WSL rsync로 처리됨.
- `rouge_score`·`scikit-learn`은 `pyproject.toml` 의존성에 없음 — 미설치 시 T2b가 조용히 skip.
- gpu03는 Blackwell(sm_120)이라 cu128+ 휠 강제였으나, 3060은 Ampere(sm_86)라 제약 없음.
- WSL RAM 15GB = 호스트의 기본 50% 배분. 상향은 `%USERPROFILE%\.wslconfig`에 `memory=24GB` 추가 후 `wsl --shutdown`.
