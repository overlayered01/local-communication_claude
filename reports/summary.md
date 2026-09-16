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