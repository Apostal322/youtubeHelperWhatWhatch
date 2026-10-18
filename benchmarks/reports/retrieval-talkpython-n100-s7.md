# Retrieval experiment: episode title → transcript (known-item)

Source https://github.com/mikeckennedy/talk-python-transcripts@accd851f0d3353bf2d3792969d7f334a4c4768b3 (youtube_transcripts/) (GPL-3.0); 100 documents; seed 7; commit `ceb0b77dfb3a112df69ba479b0189e3b54ef9dd2`.
Queries: creator episode titles from file slugs; titles withheld from documents.

> **Caveat:** titles often share rare words with their transcripts (e.g. a library name), which favours
lexical matching. This measures known-item retrieval, not topical recommendation quality.

| method | R@1 [95% CI] | R@5 | R@10 | MRR [95% CI] | nDCG@10 | median rank | false-match rate | confident false matches | query ms p50 / p95 | doc embed ms/doc |
|---|---|---|---|---|---:|---:|---:|---:|---|---:|
| hashing | 0.28 [0.1998, 0.37] | 0.63 | 0.71 | 0.4261 [0.35, 0.5049] | 0.4864 | 3.0 | 0.72 | 18 | 0.11 / 0.17 | 4.2 |
| minilm_onnx | 0.71 [0.61, 0.8] | 0.87 | 0.9 | 0.7802 [0.704, 0.8504] | 0.8056 | 1.0 | 0.29 | 14 | 2.04 / 2.52 | 805.5 |
| hybrid_rrf | 0.57 [0.47, 0.67] | 0.81 | 0.84 | 0.6734 [0.5937, 0.7487] | 0.7074 | 1.0 | 0.43 | — | — / — | — |

Similarity distributions (cosine):

* hashing: {'relevant_median': 0.1135, 'non_relevant_median': 0.0, 'non_relevant_p95': 0.125}
* minilm_onnx: {'relevant_median': 0.4779, 'non_relevant_median': 0.1975, 'non_relevant_p95': 0.4216}
