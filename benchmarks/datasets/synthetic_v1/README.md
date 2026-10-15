# synthetic_v1 benchmark dataset

A small, fully synthetic dataset for testing and regression-checking the AI Video Discovery
evaluation and recommendation harness without any YouTube credentials.

## What it contains

- `data/fixtures/videos/<video_id>.json` (20 files): fictional videos with original transcripts.
  No real video, channel or creator exists behind any of them; channel names, games and products
  are invented. Transcripts, titles and descriptions were written for this project and are
  released under CC0-1.0. Two videos are non-English (`syn_edu_002` Spanish, `syn_howto_001` German).
  Two videos carry an `audio_fixture` profile (`noisy_clipped`, `clean`) for the audio-quality path.
- `labels.jsonl`: 280 lines, one per (video, dimension) for all 20 videos and all 14 rubric
  dimensions. Scores are integers 1-5 or `null` where a dimension cannot be judged from the
  fixture (all visual dimensions; audio quality except the two audio fixtures; factual accuracy
  for content without checkable claims; informational dimensions for pure fiction/music).
- `pairs.jsonl`: 33 pairwise preferences (`overall` or a single dimension), mostly within
  genre, some cross-genre, several pitting a hidden gem against a popular-but-shallow video.
- `personas.json`: 5 recommendation-evaluation personas with the fixture videos that are
  topically relevant and worth recommending to each (clickbait/shallow videos excluded).

## Important: these are not human annotations

**All labels are synthetic design-intent labels written by the fixture author, an AI assistant.**
They record the quality the author deliberately designed into each transcript, not the judgement
of human raters. There is a single rater (`rater_id: fixture_author_intent`,
`label_source: synthetic_design_intent`), so there is no inter-rater agreement to measure.
Use this dataset only for harness testing, smoke tests and regression tracking; do not use it to
claim the evaluator agrees with human judgement.

Popularity numbers were chosen so that they do not track quality (Spearman correlation between
mean designed quality and view count is about -0.1): hidden gems have tiny audiences,
several shallow or clickbait videos are very popular, and some excellent videos are popular too.

## Videos

| video_id | genre | lang | intended profile | subscribers | views | words |
|---|---|---|---|---:|---:|---:|
| `syn_art_001` | art | en | Hidden gem: excellent content, tiny channel | 380 | 890 | 619 |
| `syn_art_002` | art | en | Derivative, low-effort paint-along | 7,400 | 51 | 362 |
| `syn_doc_001` | documentary | en | Hidden gem: excellent content, tiny channel | 920 | 2,870 | 678 |
| `syn_doc_002` | documentary | en | Clickbait: title promises a discovery the vlog never delivers | 2,400,000 | 3,310,000 | 416 |
| `syn_edu_001` | education | en | Hidden gem; excellent lecture, poor recording (noisy_clipped audio fixture) | 340 | 1,180 | 706 |
| `syn_edu_002` | education | es | Average / mixed | 9,800 | 96 | 422 |
| `syn_ent_001` | entertainment | en | Popular but shallow: filler, hype, little information | 15,300,000 | 12,100,000 | 389 |
| `syn_ent_002` | entertainment | en | Original, well-crafted fiction | 1,240,000 | 9,300,000 | 518 |
| `syn_game_001` | gaming | en | Creative narrative let's-play, tiny channel | 1,780 | 2,450 | 568 |
| `syn_game_002` | gaming | en | Clickbait (promised glitch never shown), monotone playthrough | 3,100,000 | 4,720,000 | 351 |
| `syn_howto_001` | howto | de | Solid, well-structured tutorial | 520,000 | 3,600,000 | 486 |
| `syn_howto_002` | howto | en | Popular but shallow: filler, hype, little information | 18,200,000 | 14,500,000 | 343 |
| `syn_music_001` | music | en | Original song, modest production | 96,000 | 5,200,000 | 356 |
| `syn_news_001` | news_commentary | en | Average commentary, somewhat one-sided | 6,300 | 75 | 528 |
| `syn_sci_001` | science | en | Hidden gem: excellent content, tiny channel | 640 | 2,210 | 580 |
| `syn_sci_002` | science | en | Excellent and popular (clean audio fixture) | 4,200,000 | 10,800,000 | 656 |
| `syn_sci_003` | science | en | Decent explainer with a confidently stated factual error | 1,350,000 | 690,000 | 489 |
| `syn_tech_001` | technology | en | Hidden gem: excellent content, tiny channel | 410 | 1,630 | 688 |
| `syn_tech_002` | technology | en | Popular but shallow: filler, hype, little information | 12,400,000 | 8,350,000 | 457 |
| `syn_tech_003` | technology | en | Average / mixed | 14,200 | 140 | 480 |
