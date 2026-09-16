# llmcomm — 로컬 LLM / TTS / STT 옵션 검증 테스트베드

자연스러운 한국어 음성 대화 시스템을 만들기 전에, 후보 옵션들을 **같은 조건에서 측정하고 비교**하기 위한 프로젝트입니다.
여기서 나온 수치와 청취 결과를 근거로 실제 프로젝트의 구성 요소를 확정합니다.

## 구조

```
configs/            옵션 정의(YAML). 파일 하나 = 비교 대상 하나
  llm/ tts/ stt/
data/prompts/       평가용 한국어 프롬프트 세트 (jsonl)
src/llmcomm/
  core/             공통 인터페이스(LLMBackend, TTSEngine, STTEngine), 레지스트리, VRAM 모니터
  llm/              Ollama, OpenAI 호환 서버(vLLM/llama.cpp/LM Studio), llama.cpp in-process
  tts/              Supertonic, MeloTTS, CosyVoice2, Edge(클라우드 기준점), Windows SAPI(하한 기준점)
  stt/              faster-whisper
  pipeline/         스트리밍 문장 분리 → TTS 큐 → 순서 보장 재생
  bench/            벤치마크 러너, 지표(CER 등), 마크다운 리포트
  server/           FastAPI WebSocket 서버 + 웹 테스터
reports/            results.jsonl (누적), summary.md, audio/<option>/*.wav
```

새 옵션 추가 = 어댑터 클래스 1개 + `core/registry.py` 등록 + YAML 1개.

## 설치

```powershell
uv sync --extra dev --extra whisper --extra edge --extra supertonic --extra melotts
uv run python -m unidic download          # MeloTTS 일본어 사전(한국어에도 import 시 필요)
# torch는 pyproject의 [tool.uv.sources]로 CUDA 12.8 빌드가 고정됩니다 (RTX 50xx 필수).
# Supertonic 가중치는 첫 실행 시 models/supertonic-assets/ 로 자동 다운로드됩니다.
# CosyVoice2는 별도 저장소 클론이 필요합니다. src/llmcomm/tts/cosyvoice.py docstring 참고.
```

Ollama 모델:

```powershell
ollama pull qwen3:14b; ollama pull qwen3:8b; ollama pull exaone3.5:7.8b; ollama pull gemma3:12b
```

## 사용

```powershell
uv run llmcomm list
uv run llmcomm chat --llm ollama_qwen3_14b                       # 텍스트 대화
uv run llmcomm chat --llm ollama_qwen3_14b --tts sapi_default    # 문장 단위 음성 출력
uv run llmcomm bench llm ollama_qwen3_14b ollama_exaone35_7b --repeats 3
uv run llmcomm bench tts sapi_default melotts_kr --stt whisper_turbo
uv run llmcomm bench e2e ollama_qwen3_14b melotts_kr
uv run llmcomm bench llm ollama_qwen3_8b --prompts korean_multiturn  # 문맥 유지 평가
uv run llmcomm judge --judge ollama_qwen3_14b                    # LLM 답변 품질 채점 (1~5)
uv run llmcomm report                                            # reports/summary.md 생성
uv run llmcomm serve                                             # http://127.0.0.1:8080 웹 테스터
```

## 측정 지표

| 대상 | 지표 | 의미 |
|---|---|---|
| LLM | ttft_ms | 첫 토큰까지 시간. 대화 반응성의 핵심 |
| LLM | tokens_per_s_est | 생성 속도 (토큰 수는 글자 기반 추정) |
| LLM | vram_used_mb | 실행 중 GPU 메모리 피크 (전체 GPU 기준) |
| TTS | rtf | 합성 시간 / 오디오 길이. 1 미만이면 실시간 |
| TTS | ms_per_char | 글자당 합성 시간. 문장 길이 보정 비교용 |
| TTS | roundtrip_cer | STT로 되돌린 텍스트의 문자 오류율. 명료도 대리 지표 |
| E2E | first_audio_ms | 사용자 입력 후 첫 음성이 나오기까지. 체감 지연 |

정성 평가(자연스러움, 말투, 목소리 호감도)는 `reports/audio/`의 WAV를 청취하고 별도 시트에 기록합니다.
LLM 답변 품질은 `llmcomm judge`로 자연스러움, 한국어 정확성, 관련성, 간결성을 1~5점 채점합니다.
채점 모델이 후보 중 하나면 자기 선호 편향이 생기므로, 가능하면 후보가 아닌 더 큰 모델을 judge로 씁니다.

## 비교 시 주의

- 같은 GPU에 다른 모델이 올라가 있으면 VRAM 수치가 오염됩니다. 한 번에 한 옵션만 띄웁니다.
- warmup 1회 후 측정하지만, 첫 프롬프트는 여전히 느릴 수 있어 `--repeats 3` 이상을 권합니다.
- Edge TTS는 배포 후보가 아니라 "클라우드 품질은 이 정도" 기준점입니다.
- Qwen3 계열은 `think: false`로 사고 모드를 꺼야 대화 지연이 공정하게 비교됩니다.
- Supertonic PyPI 패키지는 CPU 프로바이더만 쓰도록 되어 있어, `device: cuda` 설정 시 어댑터가 프로바이더 목록을 교체합니다. onnxruntime-gpu가 필요합니다.

## Windows 환경 이슈와 해결 (검증 중 확인)

| 문제 | 원인 | 해결 (코드에 반영됨) |
|---|---|---|
| torch가 GPU를 못 봄 | PyPI 기본 휠이 CPU 전용 | `[tool.uv.sources]`로 cu128 인덱스 고정 |
| faster-whisper `cublas64_12.dll not found` | ctranslate2가 CUDA DLL을 못 찾음 | `core/gpu.py: register_cuda_dlls()`가 torch/lib를 DLL 경로에 등록 |
| onnxruntime-gpu가 조용히 CPU로 폴백 | 같은 DLL 문제 | 같은 헬퍼 호출 + CUDA 세션 여부를 검증해 실패 시 예외 |
| MeloTTS 한국어: `eunjeon` 설치 실패 | Visual Studio 빌드 필요 | `shims/eunjeon`: Kiwi 기반 `Mecab.pos()` 대체 구현 |
| MeloTTS import 깨짐 | `python-mecab-ko`의 `mecab/`가 NTFS에서 `MeCab/`(mecab-python3)와 충돌 | python-mecab-ko를 설치하지 않음. Kiwi shim 사용 |
| Ollama 다음 모델 VRAM 오염 | 이전 모델이 keep_alive로 GPU에 잔류 | `close()`에서 keep_alive=0 후 `/api/ps` 폴링으로 언로드 확인 |
| VRAM 절대값이 6GB부터 시작 | 데스크톱/브라우저 사용량 | 유휴 기준선 대비 `vram_delta_mb` 기록 |
| onnxruntime-gpu `cublasLt64_13.dll` 없음 | ORT 1.30은 CUDA 13 빌드, torch cu128은 CUDA 12 DLL만 번들 | `nvidia-cublas`, `nvidia-cuda-runtime`, `nvidia-cudnn-cu13` 등 pip 런타임 추가 (supertonic extra) |
| E2E에서 한 번씩 5초 지연 | kss가 첫 flush 시점에 지연 import | 모듈 로드 시 import |
