# llmcomm — 로컬 LLM / TTS / STT 옵션 검증 테스트베드

자연스러운 한국어 음성 대화 시스템을 만들기 전에, 후보 옵션들을 **같은 조건에서 측정하고 비교**하기 위한 프로젝트입니다.
여기서 나온 수치와 청취 결과를 근거로 실제 프로젝트의 구성 요소를 확정합니다.

## 구조

```
configs/            옵션 정의(YAML). 파일 하나 = 비교 대상 하나
  llm/ tts/ stt/ prompt/ rag/   (prompt = 시스템 프롬프트 옵션, rag = 코퍼스 + 임베딩 + 검색 방식 조합)
data/prompts/       평가용 한국어 프롬프트 세트 (jsonl)
src/llmcomm/
  core/             공통 인터페이스(LLMBackend, TTSEngine, STTEngine), 레지스트리, VRAM 모니터
  llm/              Ollama, OpenAI 호환 서버(vLLM/llama.cpp/LM Studio), llama.cpp in-process
  tts/              Supertonic, MeloTTS, CosyVoice2, Edge(클라우드 기준점), Windows SAPI(하한 기준점)
  stt/              faster-whisper(GPU/CPU, 한국어 파인튠 CT2 변환 포함), sherpa-onnx(SenseVoice, 한국어 zipformer 오프라인/스트리밍), Vosk(하한 기준점)
  pipeline/         스트리밍 문장 분리 → TTS 큐 → 순서 보장 재생
  bench/            벤치마크 러너, 지표(CER 등), 마크다운 리포트
  server/           FastAPI WebSocket 서버 + 웹 테스터 (옵션 로드/warmup 상태, 단계별 지연, 문장별 합성·재생 상태)
data/rag/<corpus>/  RAG 코퍼스(md). 샘플 3종: product_faq(제품 FAQ), user_memory(사용자 기억), persona_lore(캐릭터 설정)
data/stt/           STT 평가 세트: 매니페스트(jsonl) + audio/<voice>/*.wav, 웹 테스터 마이크 녹음은 audio/mic/ + mic_log.jsonl
reports/            results.jsonl (누적), summary.md, audio/<option>/*.wav
```

새 옵션 추가 = 어댑터 클래스 1개 + `core/registry.py` 등록 + YAML 1개.

## 시스템 프롬프트 옵션

`configs/prompt/<name>.yaml`은 페르소나 텍스트와 `style` 규칙(말투 반말/존댓말, 최대 문장 수, 이모지·마크다운 금지, 한국어만)으로 구성됩니다.
규칙은 시스템 프롬프트에 지시문으로 자동 추가되고, 같은 규칙으로 응답을 기계 채점합니다(`n_sentences`, `emoji_count`, `polite/casual_sentences`, `style_violations`).
프리셋: `default`(1차 기준선, 규칙 없음), `voice_banmal`, `voice_jondae`, `persona_friend`(few-shot 포함), `assistant_polite`, `one_liner`.

```powershell
uv run llmcomm prompt voice_banmal                                    # 합성된 시스템 텍스트 확인
uv run llmcomm bench llm ollama_qwen3_8b --prompt default --prompt voice_banmal --prompt voice_jondae
uv run llmcomm bench e2e ollama_qwen3_8b supertonic_gpu --prompt voice_banmal
uv run llmcomm chat --llm ollama_qwen3_8b --prompt persona_friend
```

기본이 아닌 프롬프트로 측정한 행은 `ollama_qwen3_8b@prompt=voice_banmal`처럼 분리되고, judge에는 사용된 시스템 프롬프트가 함께 전달되어 말투 일관성을 지시 기준으로 채점합니다.
웹 테스터의 "시스템 프롬프트" 패널은 페르소나·말투·문장 수·금지 규칙·few-shot을 폼으로 편집합니다. 편집한 상태로 전송하면 저장 없이 적용되어 결과에 `이름*`로 표시되고, "프리셋 저장"으로 새 YAML을 만들 수 있습니다. 응답마다 규칙 위반 수가 표시됩니다.

## RAG 옵션

`configs/rag/<name>.yaml` = 코퍼스 + 임베더(Ollama 임베딩 모델 또는 없음) + 검색 방식(dense / BM25 / 하이브리드 RRF) + top_k + 컨텍스트 머리말.
검색된 청크는 시스템 프롬프트 뒤에 붙습니다. RAG 대상 3종을 모두 옵션으로 두었습니다: 제품 FAQ(정답 있는 도메인 지식), 사용자 기억(개인화), 페르소나 설정(캐릭터 1인칭 세계관).
평가 세트 `data/prompts/korean_rag_<corpus>.jsonl`은 질문마다 정답 섹션(`gold_source`)과 답에 포함돼야 하는 키워드(`must_include_any`)를 갖습니다. 자료에 없는 질문도 넣어 "모른다"고 답하는지 봅니다.

```powershell
ollama pull bge-m3; ollama pull qwen3-embedding:0.6b; ollama pull nomic-embed-text
uv run llmcomm rag faq_bge_m3 --query "루미 링이 보라색인데?"      # 검색 결과 확인
uv run llmcomm bench rag faq_bge_m3 faq_bm25 faq_bge_m3_hybrid --llm ollama_qwen3_8b --prompt voice_jondae
```

지표: `recall`(정답 섹션이 top_k에 포함), `keyword_hit`(답에 키워드 포함), `retrieval_ms`, `context_chars`, `ttft_ms`(프롬프트가 길어진 비용).
RAG 없는 대조 실행(`@rag=none:<corpus>`)이 자동으로 함께 기록됩니다. 임베딩은 `models/rag/`에 캐시됩니다.
웹 테스터의 RAG 드롭다운을 켜면 "검색된 근거" 패널에 청크와 점수, 검색 시간이 표시됩니다. "RAG 설정" 패널에서 코퍼스, 임베딩 모델(Ollama에 받은 것 자동 목록), top_k, 청크 크기, 하이브리드, 지시문을 바꿔 바로 적용(`이름*`)하거나 프리셋으로 저장할 수 있습니다. 설정이 바뀌면 다음 요청에서 다시 인덱싱합니다.

## 세부 파라미터 조정

YAML은 **프리셋**이고, 그 위에 실행 시점 오버라이드를 얹습니다. 어댑터마다 `PARAMS`로 조정 가능한 파라미터(이름, 타입, 범위, 재로드 필요 여부)를 선언하고,
결과 행에는 바꾼 값이 자동으로 붙습니다(`ollama_qwen3_8b{temperature=0.3,num_ctx=4096}`). results.jsonl의 `meta.params`에 실제 적용된 전체 파라미터가 남습니다.

```powershell
uv run llmcomm params llm ollama_qwen3_8b                                  # 조정 가능한 파라미터 목록
uv run llmcomm bench llm ollama_qwen3_8b --set temperature=0.3 --set num_ctx=4096
uv run llmcomm bench tts supertonic_gpu --sweep total_steps=4,6,8            # 한 파라미터를 여러 값으로
uv run llmcomm bench stt whisper_turbo --sweep beam_size=1,3,5
uv run llmcomm bench e2e ollama_qwen3_8b supertonic_gpu --set-llm temperature=0.3 --set-tts voice=M1 --set-pipeline min_chars=10
```

웹 테스터의 "세부 옵션" 패널은 선택한 옵션의 스키마로 슬라이더/토글을 자동 생성합니다. 값을 바꾸면 요청마다 적용되고,
`재로드` 표시가 있는 항목(모델, 장치 등)은 엔진을 다시 올립니다. "프리셋 저장"으로 현재 값을 새 YAML로 만들 수 있습니다.
파이프라인 자체의 파라미터(`tts_concurrency`, `min_chars`)도 같은 방식으로 조정합니다.

## 설치

```powershell
uv sync --extra dev --extra gpu --extra whisper --extra edge --extra supertonic --extra melotts --extra stt
uv run python -m unidic download          # MeloTTS 일본어 사전(한국어에도 import 시 필요)
# torch는 pyproject의 [tool.uv.sources]로 CUDA 12.8 빌드가 고정됩니다 (RTX 50xx 필수).
# Supertonic 가중치는 첫 실행 시 models/supertonic-assets/ 로 자동 다운로드됩니다.
# CosyVoice2는 별도 저장소 클론이 필요합니다. src/llmcomm/tts/cosyvoice.py docstring 참고.
# STT 모델(sherpa-onnx, vosk)은 첫 실행 시 models/ 아래로 자동 다운로드됩니다. Whisper는 Hugging Face 캐시를 씁니다.
# 한국어 파인튠 Whisper는 CTranslate2로 변환해야 합니다:  uv run llmcomm stt-convert ghost613/whisper-large-v3-turbo-korean
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
uv run llmcomm make-stt-set --tts edge_sunhi --tts supertonic_gpu # STT 평가 세트 합성 (data/stt/korean_stt.jsonl)
uv run llmcomm bench stt sensevoice_small whisper_turbo zipformer_ko  # STT 지연·RTF·CER·VRAM
uv run llmcomm report                                            # reports/summary.md 생성
uv run llmcomm serve                                             # http://127.0.0.1:8080 웹 테스터 (진행 단계·문장별 합성/재생 상태 표시)
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
| STT | latency_ms / rtf | 발화 전체를 인식하는 시간과 음성 길이 대비 비율. 발화 종료 후 LLM으로 넘어가기까지의 지연 |
| STT | cer | 정답 대비 문자 오류율(공백·문장부호 무시) |

정성 평가(자연스러움, 말투, 목소리 호감도)는 `reports/audio/`의 WAV를 청취하고 별도 시트에 기록합니다.
LLM 답변 품질은 `llmcomm judge`로 자연스러움, 한국어 정확성, 관련성, 간결성을 1~5점 채점합니다.
채점 모델이 후보 중 하나면 자기 선호 편향이 생기므로, 가능하면 후보가 아닌 더 큰 모델을 judge로 씁니다.

## STT 후보

| option | 엔진 | 장치 | 스트리밍 | 용도 |
|---|---|---|---|---|
| whisper_turbo / whisper_turbo_int8 | faster-whisper large-v3-turbo | GPU | ✕ | 기준 정확도. int8은 VRAM 절약 |
| whisper_large_v3 | faster-whisper large-v3 | GPU | ✕ | 정확도 상한 |
| whisper_medium / whisper_small | faster-whisper | GPU | ✕ | 크기 대비 정확도 |
| whisper_small_cpu | faster-whisper small int8 | CPU | ✕ | GPU 경합 없는 Whisper |
| whisper_turbo_ko | ghost613 한국어 파인튠 turbo (CT2 변환) | GPU | ✕ | 한국어 특화 효과 확인 |
| sensevoice_small | FunAudioLLM SenseVoice-Small (sherpa-onnx int8) | CPU | ✕ | 매우 빠름, 다국어 |
| zipformer_ko | k2-fsa 한국어 zipformer (sherpa-onnx) | CPU | ✕ | 한국어 전용 오프라인 |
| zipformer_ko_streaming | k2-fsa 한국어 스트리밍 zipformer | CPU | ○ | 말하는 중 부분 결과. 실시간 대화용 |
| vosk_small_ko | Vosk 한국어 small | CPU | ○ | 하한 기준점 |

CPU 엔진은 LLM과 GPU를 나눠 쓰지 않는다는 점이 중요합니다(2차 분석: GPU 경합이 TTS 지연을 2배로 늘림).
평가 세트는 TTS 합성 음성이라 실제 마이크 환경의 강건성은 측정하지 않습니다. 웹 테스터에서 🎤로 말하면 녹음과 인식 결과가 `data/stt/mic_log.jsonl`에 쌓이므로, 텍스트를 교정한 뒤 사람 음성 세트로 쓰면 됩니다.

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
| sherpa-onnx가 `requested API version [28] is not available` 후 크래시 | Windows 11이 System32에 구버전 onnxruntime.dll(1.17)을 내장 → sherpa 확장 모듈이 그것을 로드 | `core/gpu.py: register_onnxruntime_dll()`이 pip onnxruntime DLL을 먼저 로드 |
| SenseVoice 2025-09-09 모델이 깨진 텍스트 출력 | sherpa-onnx 1.13.8과 모델 릴리스 불일치 | 2024-07-17 릴리스 사용 (설정에 반영) |
| uv sync가 `llmcomm.exe` 삭제 실패 | 웹 서버가 실행 중이면 런처가 잠김 | 서버 종료 후 sync |
