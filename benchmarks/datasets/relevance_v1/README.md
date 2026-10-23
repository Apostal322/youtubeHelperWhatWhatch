# relevance_v1: query–video relevance benchmark (protocol; no judgments yet)

Until now, the recommendation benchmarks used a *proxy*: an episode counted as relevant to a
persona when it came from the matching podcast show. That measures show membership, not whether
a person looking for a topic would want the episode. This dataset replaces the proxy with human
judgments. **None have been collected.** `avd bench-relevance` reports `blocked_missing_labels`,
and no recommendation improvement is claimed from the proxy labels.

* `queries.jsonl`: 10 information needs, written on 2026-10-19 **before** any retrieval result
  for them was looked at. Each records language, intended audience and desired depth.
* Items: the `real_pilot_v1` **dev** split (10 episodes). The test split and `real_holdout_v2` are
  never judged here, so they stay usable for confirmation later.
* Grades: 0 = not relevant; 1 = partly relevant (touches the topic); 2 = clearly relevant (a
  person with this need would want it). "Can't tell" is not a grade and is not stored.

## Query audit (2026-10-20)

| Issue | Queries | Consequence |
|---|---|---|
| **Written from the catalog's show topics.** The author knew the dev split's show-level topic labels (not the transcripts), and 8 of 10 queries map onto one show each: Go Time q01, q02; Practical AI q03, q04, q09; Founders Talk q05, q06; Backstage q07 | 8 of 10 | Not independent of the catalog: lexical overlap with show topics favours lexical retrieval, and judged relevance may still track show membership. No result was tuned on them, but they cannot show that retrieval generalises |
| **Cross-show needs.** No show is about these topics, so relevant items, if any, are scattered | q08 (open source sustainability), q10 (career advice) | The useful test of topical retrieval. They may have no relevant item; such queries are counted (`queries_without_relevant_item`) and left out of the nDCG and Recall means |
| **Conjunctions** | q02 ("testing *and* error handling"), q06 ("stories *and* burnout") | Judge 2 only if both parts are covered, 1 if one part is |
| **Broad** | q04, q09 | Expect lower agreement; agreement is reported per benchmark run |
| **Depth and audience** | all | Recorded with each query, but graded relevance is topical only; depth fit is part of the quality rubric |

Before the benchmark is used to compare retrievers, add queries written by raters **who have not
seen the catalog**. Each rater writes one information need before receiving a packet. Keep both
sets, and report them separately.

**How (since 2026-10-20).** Every query carries a `source`: the 10 above are
`catalog_aware_author`. Send each volunteer
[`docs/templates/query-elicitation.txt`](../../../docs/templates/query-elicitation.txt) before
any packet or episode list, and add their answer as the next `qNN` with
`"source": "blind_rater"`, unedited. New queries reach packets issued after they are added.
`avd bench-relevance` reports `results` over all judged queries and `results_by_query_source`
for each source. Only the `blind_rater` results say anything about retrieval beyond these
shows' topics; with a handful of queries they are anecdotes, not estimates.

## Collection

Relevance is judged by the same volunteers who rate quality, at the end of each development item
in the offline rating packet ([`docs/annotation-pilot.md`](../../../docs/annotation-pilot.md)). A
rater who has just read the whole transcript can grade all 10 queries for it in a few minutes.
Judging a 10,000-word episode for relevance without reading it would be guesswork.

* Raters see the queries, never any retrieval or recommendation output for them.
* Calibration and test items carry no relevance questions.
* `avd annotate export` writes `relevance.jsonl` in the format below. A CSV sheet for other
  collection methods is available from `avd bench-relevance sheet`.

```json
{"query_id": "q01", "video_id": "chl_gotime_47", "rater": "volunteer-a", "grade": 2}
```

## Analysis

`avd bench-relevance --judgments relevance.jsonl`:

1. Reports agreement first: ordinal Krippendorff's alpha, raters per pair, and every pair whose
   grades span 0 to 2 (`disputed_pairs`, kept and listed, not averaged away). The pooled grade
   is the median.
2. For each of these rankings:
   * lexical retrieval (hashing);
   * MiniLM;
   * hybrid score fusion of both;
   * the **final recommendation list** for a user whose only interest is the query.

   It reports nDCG@5, Recall@5, precision@5 (grade ≥ 1), MRR and the off-topic rate (grade 0 in
   the top 5).
3. Needs at least 3 raters on every pair before rankings are compared. With fewer, it warns.

Ten items cannot rank retrievers reliably. The dataset exists to replace the proxy and to make
the protocol concrete. A meaningful comparison needs more items, and queries independent of the
catalog.
