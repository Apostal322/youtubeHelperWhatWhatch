# Recommendation stage analysis: real_pilot@v1 dev (exploratory; development data only)

Commit `17fc8179d8569d8c8528636fba25da9feef2d38c`. Labels: topical proxy: persona's matching publisher show (not human relevance judgements). Quality: heuristic baseline (no AI model available); quality features are not AI judgements.
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

## Full recommender, diversity 0.3

| embedder | recall@5 | precision@5 | MRR | nDCG@5 | returned | exploration slots | relevant items' fate |
|---|---:|---:|---:|---:|---:|---:|---|
| hashing | 0.5833 | 0.3 | 0.875 | 0.6092 | 3.5 | 1.0 | below_relevance_floor 4, selected 6 |
| minilm_chunks | 0.7917 | 0.4 | 0.875 | 0.7881 | 4.25 | 1.0 | channel_cap 1, not_selected 1, selected 8 |
| minilm_keywords | 0.7917 | 0.4 | 0.875 | 0.7646 | 4.25 | 1.0 | channel_cap 1, not_selected 1, selected 8 |
