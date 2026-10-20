# Recommendation stage analysis: real_pilot@v1 dev (exploratory; development data only)

Commit `f59c3c5b44c921243a0a61d99b3f9bedf51d9c95`. Labels: topical proxy: persona's matching publisher show (not human relevance judgements). Quality: heuristic baseline (no AI model available); quality features are not AI judgements.
4 personas; small n: read directions, not decimals.

## Retrieval only (similarity over the identical candidate pool)

| embedder | recall@5 | precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| hashing | 1.0 | 0.5 | 0.8333 | 0.8485 |
| minilm_chunks | 1.0 | 0.5 | 0.875 | 0.906 |
| minilm_keywords | 1.0 | 0.5 | 0.8333 | 0.8559 |
| hybrid(hashing+minilm_chunks) (score fusion) | 1.0 | 0.5 | 0.875 | 0.9127 |
| hybrid(hashing+minilm_keywords) (score fusion) | 1.0 | 0.5 | 0.875 | 0.9127 |

## Full recommender, diversity 0.0

| embedder | recall@5 | precision@5 | MRR | nDCG@5 | returned | exploration slots | relevant items' fate |
|---|---:|---:|---:|---:|---:|---:|---|
| hashing | 0.5833 | 0.3 | 0.8333 | 0.5891 | 3.5 | 1.0 | below_relevance_floor 4, selected 6 |
| minilm_chunks | 0.7917 | 0.4 | 0.875 | 0.7881 | 4.25 | 1.0 | channel_cap 1, not_selected 1, selected 8 |
| minilm_keywords | 0.9167 | 0.45 | 0.8333 | 0.8105 | 4.25 | 1.0 | channel_cap 1, selected 9 |

Selection policies: `two_stage_mmr_r2:651dd4268a32`, `two_stage_mmr_r2:b9d27ef50cc5`, `two_stage_mmr_r2:c2ec3317ea49`.

## Full recommender, diversity 0.3

| embedder | recall@5 | precision@5 | MRR | nDCG@5 | returned | exploration slots | relevant items' fate |
|---|---:|---:|---:|---:|---:|---:|---|
| hashing | 0.5833 | 0.3 | 0.875 | 0.6092 | 3.5 | 1.0 | below_relevance_floor 4, selected 6 |
| minilm_chunks | 0.7917 | 0.4 | 0.875 | 0.7881 | 4.25 | 1.0 | channel_cap 1, not_selected 1, selected 8 |
| minilm_keywords | 0.7917 | 0.4 | 0.875 | 0.7646 | 4.25 | 1.0 | channel_cap 1, not_selected 1, selected 8 |

Selection policies: `two_stage_mmr_r2:651dd4268a32`, `two_stage_mmr_r2:b9d27ef50cc5`, `two_stage_mmr_r2:c2ec3317ea49`.

## Why each candidate was shown or not (minilm_chunks, diversity 0.3)

Ordered by similarity. `proxy` = from the persona's show (the proxy label, not a human judgement).

| persona | item | proxy | similarity | calibrated | fate | position |
|---|---|:-:|---:|---:|---|---:|
| gopher | chl_gotime_226 | yes | 0.4687 | 0.8956 | selected | 1 |
| gopher | chl_gotime_179 | yes | 0.4057 | 0.6858 | channel_cap |  |
| gopher | chl_gotime_47 | yes | 0.4005 | 0.6684 | selected | 2 |
| gopher | chl_practicalai_178 |  | 0.3069 | 0.3563 | not_selected |  |
| gopher | chl_practicalai_65 |  | 0.3058 | 0.3526 | selected | 3 |
| gopher | chl_practicalai_100 |  | 0.2825 | 0.275 | not_selected |  |
| gopher | chl_founderstalk_58 |  | 0.262 | 0.2065 | selected | 4 |
| gopher | chl_founderstalk_76 |  | 0.2123 | 0.041 | not_selected |  |
| gopher | chl_backstage_21 |  | 0.2022 | 0.0075 | not_selected |  |
| gopher | chl_backstage_4 |  | 0.1875 | 0.0 | selected_exploration | 5 |
| ml_practitioner | chl_practicalai_65 | yes | 0.5385 | 1.0 | selected | 1 |
| ml_practitioner | chl_practicalai_100 | yes | 0.5167 | 1.0 | selected | 2 |
| ml_practitioner | chl_practicalai_178 | yes | 0.4735 | 0.9116 | selected | 3 |
| ml_practitioner | chl_gotime_226 |  | 0.0962 | 0.0 | selected_exploration | 4 |
| ml_practitioner | chl_backstage_21 |  | 0.0817 | 0.0 | below_relevance_floor |  |
| ml_practitioner | chl_backstage_4 |  | 0.0683 | 0.0 | below_relevance_floor |  |
| ml_practitioner | chl_gotime_179 |  | 0.066 | 0.0 | below_relevance_floor |  |
| ml_practitioner | chl_gotime_47 |  | 0.0507 | 0.0 | below_relevance_floor |  |
| ml_practitioner | chl_founderstalk_58 |  | 0.0446 | 0.0 | below_relevance_floor |  |
| ml_practitioner | chl_founderstalk_76 |  | 0.0322 | 0.0 | below_relevance_floor |  |
| founder | chl_founderstalk_76 | yes | 0.5731 | 1.0 | selected | 1 |
| founder | chl_founderstalk_58 | yes | 0.5419 | 1.0 | selected | 2 |
| founder | chl_backstage_4 |  | 0.1644 | 0.0 | below_relevance_floor |  |
| founder | chl_gotime_47 |  | 0.086 | 0.0 | below_relevance_floor |  |
| founder | chl_backstage_21 |  | 0.0858 | 0.0 | below_relevance_floor |  |
| founder | chl_practicalai_100 |  | 0.0512 | 0.0 | below_relevance_floor |  |
| founder | chl_practicalai_65 |  | 0.041 | 0.0 | below_relevance_floor |  |
| founder | chl_practicalai_178 |  | 0.0299 | 0.0 | below_relevance_floor |  |
| founder | chl_gotime_226 |  | 0.0123 | 0.0 | selected_exploration | 3 |
| founder | chl_gotime_179 |  | -0.0019 | 0.0 | below_relevance_floor |  |
| podcast_insider | chl_founderstalk_76 |  | 0.3969 | 0.6563 | selected | 1 |
| podcast_insider | chl_backstage_4 | yes | 0.366 | 0.5535 | selected | 2 |
| podcast_insider | chl_founderstalk_58 |  | 0.3639 | 0.5464 | selected | 3 |
| podcast_insider | chl_practicalai_100 |  | 0.2439 | 0.1462 | selected | 4 |
| podcast_insider | chl_backstage_21 | yes | 0.2233 | 0.0778 | not_selected |  |
| podcast_insider | chl_gotime_47 |  | 0.1691 | 0.0 | below_relevance_floor |  |
| podcast_insider | chl_practicalai_65 |  | 0.1297 | 0.0 | below_relevance_floor |  |
| podcast_insider | chl_practicalai_178 |  | 0.1282 | 0.0 | below_relevance_floor |  |
| podcast_insider | chl_gotime_179 |  | 0.0685 | 0.0 | below_relevance_floor |  |
| podcast_insider | chl_gotime_226 |  | 0.0482 | 0.0 | selected_exploration | 5 |
