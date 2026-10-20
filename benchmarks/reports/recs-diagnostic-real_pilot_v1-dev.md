# Why do the embedders disagree? Diagnostic on real_pilot@v1 dev (exploratory; informs no decision)

Commit `8cf4edd1c9b523f144d9764acb1180dc173002ac`. Labels: topical proxy: persona's matching publisher show. 4 personas; tiny n, exploratory: no default is changed on this basis.

| embedder | recipe | nDCG@5 full recommender | nDCG@5 similarity only | items returned (full) |
|---|---|---:|---:|---:|
| hashing | meta_keywords_v1 | 0.5386 | 0.8485 | 3.25 |
| minilm_chunks | chunk_mean_v1 | 0.7294 | 0.906 | 4.0 |
| minilm_keywords | meta_keywords_v1 | 0.7518 | 0.8559 | 4.0 |

Mean cosine to relevant vs other items (per persona):

* hashing: gopher 0.1046 vs 0.0231; ml_practitioner 0.048 vs -0.0527; founder 0.0647 vs 0.0345; podcast_insider 0.074 vs 0.0201
* minilm_chunks: gopher 0.425 vs 0.2513; ml_practitioner 0.5096 vs 0.0628; founder 0.5575 vs 0.0586; podcast_insider 0.2947 vs 0.1936
* minilm_keywords: gopher 0.3093 vs 0.2334; ml_practitioner 0.4837 vs 0.0339; founder 0.5191 vs 0.0657; podcast_insider 0.26 vs 0.208
