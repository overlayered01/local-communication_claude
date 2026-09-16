# Benchmark summary

median per option; latency columns also show p95 in parentheses

## LLM

| option | n | ttft_ms | tokens_per_s_est | total_ms | vram_delta_mb |
|---|---|---|---|---|---|
| ollama_exaone35_7b | 24 | 284 (306) | 196.9 | 545 (691) | 5874 |
| ollama_gemma3_12b | 24 | 329 (380) | 93.8 | 519 (750) | 9597 |
| ollama_llama31_8b | 24 | 280 (301) | 147.6 | 388 (529) | 5957 |
| ollama_qwen3_14b | 24 | 291 (326) | 79.6 | 602 (1033) | 10325 |
| ollama_qwen3_8b | 24 | 286 (316) | 127.9 | 482 (907) | 6352 |

## TTS

| option | n | total_ms | rtf | ms_per_char | roundtrip_cer | vram_delta_mb |
|---|---|---|---|---|---|---|
| edge_sunhi | 8 | 563 (910) | 0.1 | 15.8 | 0 | 2152 |
| melotts_kr | 8 | 298 (628) | 0.0 | 9 | 0 | 981 |
| sapi_default | 8 | 463 (490) | 0.1 | 14.6 | 0 | 2152 |
| supertonic | 8 | 2075 (2768) | 0.4 | 61.9 | 0 | 2152 |
| supertonic_gpu | 8 | 195 (215) | 0.0 | 6.0 | 0 | 1368 |
| supertonic_m1_fast | 8 | 1658 (2327) | 0.3 | 47.9 | 0 | 2152 |

## E2E

| option | n | ttft_ms | first_sentence_ms | first_audio_ms | llm_done_ms | total_ms | audio_sec | vram_delta_mb |
|---|---|---|---|---|---|---|---|---|
| ollama_exaone35_7b+supertonic_gpu | 8 | 282 (320) | 365 (482) | 709 (944) | 599 (785) | 1093 (1366) | 12.0 | 8092 |
| ollama_gemma3_12b+supertonic_gpu | 8 | 330 (358) | 423 (475) | 873 (973) | 536 (694) | 949 (1140) | 4.5 | 11622 |
| ollama_qwen3_14b+supertonic_gpu | 8 | 297 (331) | 424 (654) | 905 (1043) | 742 (1178) | 1123 (1464) | 7.9 | 12421 |
| ollama_qwen3_8b+melotts_kr | 8 | 284 (321) | 343 (441) | 599 (854) | 508 (890) | 696 (1317) | 7.3 | 7533 |
| ollama_qwen3_8b+supertonic_gpu | 8 | 290 (319) | 367 (415) | 827 (901) | 464 (1018) | 900 (1451) | 5.8 | 8498 |

## LLM quality (judge: ollama_qwen3_14b; 1-5, mean)

| option | n | naturalness | korean | relevance | brevity | mean |
|---|---|---|---|---|---|---|
| ollama_gemma3_12b | 24 | 4.67 | 4.96 | 4.79 | 4.83 | 4.81 |
| ollama_qwen3_8b | 24 | 4.79 | 4.88 | 4.96 | 4.58 | 4.80 |
| ollama_exaone35_7b | 24 | 4.83 | 5.00 | 4.79 | 4.21 | 4.71 |
| ollama_qwen3_14b | 24 | 4.58 | 4.88 | 4.88 | 4.38 | 4.68 |
| ollama_llama31_8b | 24 | 4.29 | 4.75 | 4.54 | 4.38 | 4.49 |