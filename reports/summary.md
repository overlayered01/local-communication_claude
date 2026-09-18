# Benchmark summary

median per option; latency columns also show p95 in parentheses

## LLM

| option | n | ttft_ms | tokens_per_s_est | total_ms | vram_delta_mb | n_sentences | emoji_count | style_violations |
|---|---|---|---|---|---|---|---|---|
| ollama_exaone35_7b | 32 | 281 (304) | 195.2 | 545 (691) | 5873 | 3.5 | 1 | 0 |
| ollama_exaone35_7b@prompt=persona_friend | 8 | 279 (312) | 183.1 | 448 (713) | 6127.5 | 2 | 0 | 0 |
| ollama_exaone35_7b@prompt=voice_banmal | 8 | 277 (298) | 185.2 | 445 (545) | 5877 | 2.5 | 0 | 0.5 |
| ollama_exaone35_7b@prompt=voice_jondae | 8 | 285 (306) | 199.8 | 546 (613) | 5879 | 3 | 0 | 1 |
| ollama_gemma3_12b | 32 | 326 (361) | 95.3 | 523 (715) | 9598 | 3 | 1 | 0 |
| ollama_gemma3_12b@prompt=persona_friend | 8 | 349 (371) | 101.6 | 681 (954) | 9634 | 3 | 0 | 0 |
| ollama_gemma3_12b@prompt=voice_banmal | 8 | 343 (366) | 92.2 | 551 (684) | 9567 | 2 | 0 | 0 |
| ollama_gemma3_12b@prompt=voice_jondae | 8 | 333 (375) | 115.3 | 754 (1146) | 9549 | 3 | 0 | 1 |
| ollama_llama31_8b | 32 | 280 (308) | 149.6 | 391 (513) | 5957 | 2 | 0 | 0 |
| ollama_llama31_8b@prompt=persona_friend | 8 | 289 (315) | 134 | 458 (576) | 5963 | 2.5 | 0 | 0 |
| ollama_llama31_8b@prompt=voice_banmal | 8 | 274 (308) | 134.3 | 409 (461) | 5973 | 2 | 0 | 0 |
| ollama_llama31_8b@prompt=voice_jondae | 8 | 276 (325) | 158.6 | 413 (690) | 5988 | 2 | 0 | 0 |
| ollama_qwen3_14b | 32 | 293 (325) | 79 | 611 (1033) | 10325 | 2.5 | 0 | 0 |
| ollama_qwen3_14b@prompt=persona_friend | 8 | 292 (344) | 76.5 | 599 (1109) | 10321 | 2.5 | 0 | 0 |
| ollama_qwen3_14b@prompt=voice_banmal | 8 | 290 (327) | 70.5 | 563 (682) | 10325 | 2 | 0 | 0 |
| ollama_qwen3_14b@prompt=voice_jondae | 8 | 292 (334) | 81.3 | 601 (891) | 10318 | 2 | 0 | 0 |
| ollama_qwen3_8b | 32 | 286 (310) | 128.8 | 476 (857) | 6352 | 3 | 0.5 | 0 |
| ollama_qwen3_8b@prompt=persona_friend | 8 | 277 (326) | 107.4 | 421 (461) | 6690 | 2 | 0 | 0 |
| ollama_qwen3_8b@prompt=voice_banmal | 8 | 281 (330) | 125.3 | 433 (515) | 6306 | 2 | 0 | 0 |
| ollama_qwen3_8b@prompt=voice_jondae | 8 | 282 (302) | 128.1 | 475 (574) | 6355 | 2 | 0 | 0 |

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

| option | n | ttft_ms | first_sentence_ms | first_audio_ms | llm_done_ms | total_ms | audio_sec | vram_delta_mb | style_violations |
|---|---|---|---|---|---|---|---|---|---|
| ollama_exaone35_7b+supertonic_gpu | 8 | 282 (320) | 365 (482) | 709 (944) | 599 (785) | 1093 (1366) | 12.0 | 8092 | - |
| ollama_exaone35_7b+supertonic_gpu@v2 | 8 | 278 (301) | 396 (454) | 693 (1015) | 546 (792) | 896 (1520) | 10.7 | 8140.5 | - |
| ollama_gemma3_12b+supertonic_gpu | 8 | 330 (358) | 423 (475) | 873 (973) | 536 (694) | 949 (1140) | 4.5 | 11622 | - |
| ollama_gemma3_12b+supertonic_gpu@v2 | 8 | 329 (355) | 416 (490) | 911 (978) | 535 (642) | 998 (1170) | 4.3 | 11667.5 | - |
| ollama_qwen3_14b+supertonic_gpu | 8 | 297 (331) | 424 (654) | 905 (1043) | 742 (1178) | 1123 (1464) | 7.9 | 12421 | - |
| ollama_qwen3_14b+supertonic_gpu@v2 | 8 | 308 (329) | 434 (588) | 936 (1090) | 949 (1746) | 1260 (1947) | 10.6 | 11834.5 | - |
| ollama_qwen3_8b+melotts_kr | 8 | 284 (321) | 343 (441) | 599 (854) | 508 (890) | 696 (1317) | 7.3 | 7533 | - |
| ollama_qwen3_8b+melotts_kr@v2 | 8 | 284 (311) | 379 (462) | 654 (848) | 514 (1046) | 737 (1400) | 8.6 | 7592.5 | - |
| ollama_qwen3_8b+supertonic_gpu | 8 | 290 (319) | 367 (415) | 827 (901) | 464 (1018) | 900 (1451) | 5.8 | 8498 | - |
| ollama_qwen3_8b+supertonic_gpu@v2 | 8 | 285 (305) | 371 (425) | 898 (950) | 518 (977) | 1110 (1296) | 6.5 | 8681 | - |

## STT

| option | n | latency_ms | rtf | cer | vram_delta_mb |
|---|---|---|---|---|---|
| sensevoice_small | 32 | 87 (119) | 0.0 | 0.03 | 0 |
| vosk_small_ko | 32 | 2404 (2699) | 0.5 | 0.16 | 106.5 |
| whisper_large_v3 | 32 | 283 (407) | 0.1 | 0.00 | 4376 |
| whisper_medium | 32 | 207 (294) | 0.0 | 0.00 | 2018 |
| whisper_small | 32 | 106 (151) | 0.0 | 0.01 | 899 |
| whisper_small_cpu | 32 | 1030 (1110) | 0.2 | 0.01 | -8.5 |
| whisper_turbo | 32 | 115 (140) | 0.0 | 0.00 | 2495 |
| whisper_turbo_int8 | 32 | 117 (157) | 0.0 | 0.00 | 1454 |
| whisper_turbo_ko | 32 | 114 (132) | 0.0 | 0.09 | 2657 |
| zipformer_ko | 32 | 66 (84) | 0.0 | 0.02 | 4 |
| zipformer_ko_streaming | 32 | 209 (298) | 0.0 | 0.03 | 0 |

## RAG

| option | n | recall | keyword_hit | retrieval_ms | context_chars | ttft_ms | total_ms | e2e_ms | style_violations |
|---|---|---|---|---|---|---|---|---|---|
| ollama_qwen3_8b@rag=faq_bge_m3@prompt=voice_jondae | 15 | 1.00 | 1.00 | 19 (2268) | 313 | 312 (338) | 544 (795) | 696 (2946) | 0 |
| ollama_qwen3_8b@rag=faq_bge_m3_hybrid@prompt=voice_jondae | 15 | 1.00 | 0.93 | 16 (21) | 357 | 315 (341) | 572 (690) | 592 (704) | 0 |
| ollama_qwen3_8b@rag=faq_bge_m3_k1@prompt=voice_jondae | 15 | 1.00 | 0.87 | 16 (18) | 112 | 298 (321) | 534 (677) | 553 (693) | 0 |
| ollama_qwen3_8b@rag=faq_bm25@prompt=voice_jondae | 15 | 1.00 | 0.93 | 0 (0) | 357 | 314 (340) | 551 (737) | 551 (737) | 0 |
| ollama_qwen3_8b@rag=faq_nomic@prompt=voice_jondae | 15 | 0.43 | 0.47 | 15 (32) | 319 | 314 (340) | 534 (660) | 555 (676) | 0 |
| ollama_qwen3_8b@rag=faq_qwen3_embed@prompt=voice_jondae | 15 | 1.00 | 0.93 | 31 (286) | 356 | 1066 (1393) | 2553 (3898) | 2796 (3915) | 0 |
| ollama_qwen3_8b@rag=lore_bge_m3@prompt=voice_jondae | 12 | 1.00 | 0.92 | 17 (20) | 218 | 306 (327) | 537 (626) | 558 (645) | 0 |
| ollama_qwen3_8b@rag=memory_bge_m3@prompt=voice_jondae | 13 | 1.00 | 0.92 | 17 (25) | 228 | 305 (323) | 514 (572) | 547 (630) | 0 |
| ollama_qwen3_8b@rag=memory_bm25@prompt=voice_jondae | 13 | 1.00 | 1.00 | 0 (0) | 220 | 308 (332) | 514 (554) | 514 (554) | 0 |
| ollama_qwen3_8b@rag=none:persona_lore@prompt=voice_jondae | 12 | 0.00 | 0.17 | 0 (0) | 0 | 302 (325) | 476 (550) | 476 (550) | 0 |
| ollama_qwen3_8b@rag=none:product_faq@prompt=voice_jondae | 90 | 0.00 | 0.12 | 0 (0) | 0 | 304 (392) | 538 (2512) | 538 (2512) | 0 |
| ollama_qwen3_8b@rag=none:user_memory@prompt=voice_jondae | 26 | 0.00 | 0.15 | 0 (0) | 0 | 298 (322) | 483 (578) | 483 (578) | 0 |

## LLM quality (judge: ollama_qwen3_14b; 1-5, mean)

| option | n | naturalness | korean | relevance | brevity | mean |
|---|---|---|---|---|---|---|
| ollama_gemma3_12b | 24 | 4.67 | 4.96 | 4.79 | 4.83 | 4.81 |
| ollama_qwen3_8b | 24 | 4.79 | 4.88 | 4.96 | 4.58 | 4.80 |
| ollama_exaone35_7b | 24 | 4.83 | 5.00 | 4.79 | 4.21 | 4.71 |
| ollama_qwen3_14b | 24 | 4.58 | 4.88 | 4.88 | 4.38 | 4.68 |
| ollama_llama31_8b | 24 | 4.29 | 4.75 | 4.54 | 4.38 | 4.49 |