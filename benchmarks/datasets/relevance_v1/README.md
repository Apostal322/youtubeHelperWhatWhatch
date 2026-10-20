# relevance_v1: query–video relevance benchmark (protocol; no judgments yet)

The recommendation benchmarks so far used a *proxy*: an episode counts as relevant to a persona
when it comes from the matching podcast show. That measures show membership, not whether a
person looking for a topic would want the episode. This dataset replaces the proxy with human
judgments.

* `queries.jsonl`: 10 information needs, written on 2026-10-19 **before** any retrieval result
  for them was looked at. Each records language, intended audience and desired depth.
* Items: the `real_pilot_v1` **dev** split (10 episodes). The test split and `real_holdout_v2` are
  never judged here, so they stay usable for confirmation later.
* `judgments.jsonl` (to be collected; **none exist yet**): one line per rater, query and item:

  ```json
  {"query_id": "q01", "video_id": "chl_gotime_47", "rater": "r-7f3a", "grade": 2,
   "language_ok": true, "depth": "matches", "audience": "matches", "minutes_spent": 6}
  ```

  `grade` is 0 = not relevant, 1 = partly relevant (touches the topic), 2 = clearly relevant
  (a person with this need would want it). `depth`/`audience` take `too_basic` | `matches` |
  `too_advanced` | `unclear`.

## Protocol

1. `avd bench-relevance sheet --out sheet.csv` lists every query × item pair. Raters read or
   listen to each episode through its publisher link or the annotation UI, never through this
   system's scores.
2. At least **3 independent raters** per pair. Raters do not see each other's grades or any model
   output. They are pseudonymous, and nothing identifying is stored.
3. Agreement is reported (Krippendorff's alpha, ordinal) before any retriever is compared. The
   pooled grade is the median.
4. `avd bench-relevance --judgments judgments.jsonl` computes graded nDCG@5, precision@5 (grade
   ≥ 1), MRR and off-topic rate (grade 0 in the top 5), for retrieval only and for the full
   recommender, per embedder.

Ten items cannot rank retrievers reliably. This dataset exists to replace the proxy and to make
the protocol concrete; a meaningful comparison needs more items.
