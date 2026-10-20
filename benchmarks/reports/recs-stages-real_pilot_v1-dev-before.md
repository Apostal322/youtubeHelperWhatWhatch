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
| hashing | 0.5 | 0.25 | 0.8333 | 0.5386 | 3.25 | 1.0 | below_relevance_floor 4, channel_cap 1, selected 5 |
| minilm_chunks | 0.7083 | 0.35 | 0.875 | 0.7294 | 4.0 | 1.0 | channel_cap 2, not_selected 1, selected 7 |
| minilm_keywords | 0.8333 | 0.4 | 0.8333 | 0.7518 | 4.0 | 1.0 | channel_cap 2, selected 8 |

## Full recommender, diversity 0.3

| embedder | recall@5 | precision@5 | MRR | nDCG@5 | returned | exploration slots | relevant items' fate |
|---|---:|---:|---:|---:|---:|---:|---|
| hashing | 0.5 | 0.25 | 0.875 | 0.5587 | 3.25 | 1.0 | below_relevance_floor 4, channel_cap 1, selected 5 |
| minilm_chunks | 0.7083 | 0.35 | 0.875 | 0.7294 | 4.0 | 1.0 | channel_cap 2, not_selected 1, selected 7 |
| minilm_keywords | 0.7083 | 0.35 | 0.875 | 0.7059 | 4.0 | 1.0 | channel_cap 2, not_selected 1, selected 7 |
