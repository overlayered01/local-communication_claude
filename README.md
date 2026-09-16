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
uv sync                       # 기본 의존성 (Python 3.11 고정)
uv sync --extra whisper       # STT 라운드트립 평가용
uv sync --extra edge          # 클라우드 기준점
# TTS 엔진은 각 어댑터 docstring의 설치 절차를 따르고 configs/tts/*.yaml 경로를 맞춥니다.
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
LLM 답변 품질은 `output_text`를 모아 강한 모델로 채점(LLM-as-judge)하거나 사람이 평가합니다.

## 비교 시 주의

- 같은 GPU에 다른 모델이 올라가 있으면 VRAM 수치가 오염됩니다. 한 번에 한 옵션만 띄웁니다.
- warmup 1회 후 측정하지만, 첫 프롬프트는 여전히 느릴 수 있어 `--repeats 3` 이상을 권합니다.
- Edge TTS는 배포 후보가 아니라 "클라우드 품질은 이 정도" 기준점입니다.
