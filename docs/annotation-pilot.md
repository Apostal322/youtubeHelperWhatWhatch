# Human annotation pilot: volunteers, no server

**Status (2026-10-20): no human rating exists.** The workflow below has been verified end to end
with scripted test input: `tests/test_annotation_offline.py`, and `scripts/e2e/annotation_packet.py`
in headless Chromium against the real pilot transcripts. **Data collection is pending:** it needs
real raters, and none have been recruited. No ratings were generated or simulated, and none will
be.

## Target

| | |
|---|---|
| Items | the 10 development episodes of `real_pilot_v1` (Changelog, CC BY-SA 4.0; all external uses confirmed) |
| Raters per item | 3, independent and blind |
| Calibration | 2 practice episodes per rater before any development item; never used as labels |
| Per dimension | a status (scored, insufficient evidence, not applicable, cannot determine, needs expertise, needs video/audio, ambiguous, abstain); a 1–5 score only when scored; confidence; citations the rater selects themselves |
| Separately | optional personal enjoyment (1–5), never used as a quality label |
| Never shown to raters | any automated assessment, other raters' work, view or subscriber counts, links to the video platform |

## Time each rater gives (an estimate, not a measurement)

The episodes are long conversational transcripts:

* the 10 development items are 9,062 to 13,768 words (109,420 in total);
* the 2 calibration items are 10,769 and 18,251 words.

Assuming careful reading at 200–250 words per minute, plus about 15 minutes to rate ten
dimensions with citations:

| Work | Estimate |
|---|---|
| One development item | 50–85 minutes |
| Calibration (both practice items) | 2.5–3 hours |
| All 10 development items | 10–12 hours |
| The whole pilot (30 ratings plus calibration for 3 raters) | about 40 rater-hours |

This is too much to ask of one volunteer, so spread it across more people. For example, **6 raters
each rating 5 items** gives every item 3 ratings, at about 6–7 hours plus calibration per person,
spread over a few weeks. Tell volunteers the honest figure. The first real packets will measure the
actual time: `avd annotate qc` reports reported durations.

## Start small: the usability pilot (15–20 minutes, never a rating)

Before asking anyone for an hour or more, test the packet itself with one or two volunteers:

```bash
avd annotate usability-packet --label pilot-1 --consent-file rating-consent.txt
# → usability-pilot-1-xxxxxxxx.html: the first ~1,500 words (whole segments) of the
#   shortest calibration episode, labelled "Usability pilot (excerpt, not a rating)"
avd annotate usability-report ratings-pilot-1-xxxxxxxx.json --consent-file rating-consent.txt
```

The report covers what the pilot is meant to find out:

| Question | What the report shows |
|---|---|
| Are the rubric and status choices understandable? | statuses chosen, unanswered dimensions, the rater's notes and per-dimension notes, `ambiguous_rubric` flags |
| Can evidence be selected accurately? | citations made, and how many were found at the cited place in the excerpt |
| Does partial work survive? Does consent work? | the return exists only after consent; drafts survive a reload (tested in the E2E) |
| How long does it take? | the time the page was open and visible, as reported by the browser |

It is kept apart from every label. Issuing it stores nothing, `import-packet` refuses its
return ("never a rating"), and the report stores nothing either. An excerpt rating must never be
compared with, or substituted for, a full-episode rating. Full episodes stay the scientific
task: calibration, then development items, as below.

**Message to send with the pilot file (template):**

> Thank you for trying this. The attached file opens in any browser and works offline: it
> cannot send anything anywhere. It holds the opening part of one podcast episode (about 1,500
> words), so expect 15–20 minutes. Please read the consent text at the top first; you can
> download your answers only after agreeing. Then rate the ten aspects as best you can, and
> write down anything that was unclear or slow, in the box at the end: that is what this trial
> is for. Your answers are never used as a rating of the episode. When you are done, press
> **Mark this item complete**, then **Download ratings file**, and send that file back to me
> privately (not in a public GitHub issue).

Also send [`templates/query-elicitation.txt`](templates/query-elicitation.txt) **before** the
first packet. It collects one search written by someone who has not seen the catalog
(`benchmarks/datasets/relevance_v1/README.md`).

## How it works without a central server

```
coordinator (has the data)                        rater (any computer, offline)
avd annotate create-rater <pseudonym>
avd annotate packet --rater <p> --consent-file c.txt ──►  packet-<p>.html (one file, opens in a browser,
                                                 no network access, progress saved locally)
                                           ◄──  ratings-<p>-xxxx.json (Download ratings file)
avd annotate import-packet ratings.json
avd annotate export --out benchmarks/labels/…
```

* **One file, no network.** The packet is a single HTML file with a content security policy that
  forbids every network request. The rater needs no account, no installation and no connection
  to anyone's server. The coordinator can send it by e-mail, a shared folder or a USB stick.
* **Blind assignment, as online:**
  * calibration first;
  * then the items with the fewest ratings, in a per-rater random order;
  * never more than 3 raters per item (assigned but unreturned items count toward the 3);
  * occasional blind repeats for test–retest, once a rater has 5 submissions.
* **Authenticated returns.** Each packet carries a random secret, stored hashed by the
  coordinator, and a return file is accepted only with it. Each item is validated on its own:
  * status rules;
  * every citation is checked against the transcript;
  * QC flags (`too_fast`, `straight_lining`, missing evidence).

  A rejected item stays assigned, so the rater can fix it and send the file again. Re-sending an
  accepted item records nothing twice.
* **Timing is self-reported.** Offline ratings carry `client_timing` (time the item was open and
  visible) and `offline_packet`, so analyses can separate them from server-timed work.
* **Consent first.** The packet shows the owner-approved consent text, and the ratings file can
  be downloaded only after the rater ticks agreement. The import refuses a file without that
  agreement to the exact text the packet was issued with. Each accepted rating records the
  text's hash (`consent:<hash>`).
* **A rater who stops.** `avd annotate release-packet <id>` frees their unreturned items for
  someone else, and keeps whatever they already returned.

## Coordinator checklist

1. **Before recruiting:** approve the consent text in `docs/templates/rating-consent.txt`. It
   is a draft: its first line, `DRAFT FOR PROJECT-OWNER APPROVAL…`, makes `avd annotate packet`
   refuse it. Adapt it, delete that line, and keep your copy. Decide where returned files are kept
   and for how long. A packet records the hash of its consent text, so changing the text later
   means reissuing unreturned packets (`release-packet`, then `packet`).
2. `avd dataset import benchmarks/datasets/real_pilot_v1/manifest.json` on the coordinator's
   machine. Blinding of model assessments is on by default.
3. For each volunteer: `avd annotate create-rater <pseudonym> --expertise <self-declared tags>`.
   Ignore the printed token; packets do not need it.
4. Optional, recommended for the first one or two volunteers: a usability pilot (above). Then
   `avd annotate packet --rater <pseudonym> --consent-file rating-consent.txt`. The first packet
   holds the 2 calibration items, shortest first. For a lighter start, `--items 1` holds only the
   shorter one (10,769 words, about 50–65 minutes). Items are always whole episodes.
   Send the file with the instructions below.
5. When a ratings file comes back: `avd annotate import-packet <file>`. Send the next packet with
   `avd annotate packet --rater <pseudonym> --items 2 --consent-file rating-consent.txt`.
6. After the calibration round, hold a short call, optional and voluntary, to discuss
   disagreements on the practice items. Do not discuss development items while they are being
   rated.
7. Weekly: `avd annotate qc`. When every development item has 3 ratings:
   `avd annotate export --out benchmarks/labels/real_pilot_v1-dev`, then
   `avd bench-real --labels benchmarks/labels/real_pilot_v1-dev`.

   **Human–human agreement is the first result.** It is reported before any comparison with a
   model, and with whatever value it has.
8. Do not change the rubric while ratings are being collected. Do not rate the test split until
   the development round is analysed.

## Invitation (template)

> **Help test whether AI can judge the quality of long-form talks: volunteers wanted**
>
> AI Video Discovery is an open-source research project. It tries to rate videos and podcasts on
> what they contain (depth, accuracy, clarity, structure, and so on) instead of how popular they
> are. Before anyone can trust such ratings, they have to be compared with careful human
> judgement, and right now we have none.
>
> We are looking for volunteers who are comfortable reading long English transcripts of
> technical podcasts (software, AI, product work). You would:
>
> * receive a single file that opens in your web browser; no account, no installation, no
>   internet connection needed while you work;
> * read two practice episodes, then a few real ones, and rate each on ten aspects using written
>   descriptions of each score. You can always say "the transcript doesn't show this" instead of
>   guessing;
> * send back the ratings file it produces.
>
> **Time:** each episode takes about 50–85 minutes to read and rate; the two practice episodes
> about 2.5–3 hours together. We ask for 5 episodes after practice, spread over a few weeks, at
> your own pace. This is unpaid volunteer work; you can stop at any time.
>
> You are identified only by a pseudonym you choose. Contact: <coordinator contact>.

## Participant instructions (template, send with the packet)

1. Open `packet-<you>.html` in Chrome, Firefox, Edge or Safari. It does not use the internet.
   Your progress is saved in that browser on that computer. Use the same browser each time, and
   do not clear its site data until you have sent your ratings.
2. Practice items are marked **[practice]**. Rate them like the others: they help everyone
   calibrate and are never counted.
3. For each of the ten aspects:
   * pick a **status** first. Choose **scored** only if you can justify a 1–5 score from the
     transcript;
   * read all five score descriptions before choosing;
   * select a few words in a transcript segment and press **Cite selection** to back your score.
     Mark a citation as *counter-evidence* if it points the other way.
4. Judge the **transcript only**: not the show, the speakers' reputation, popularity or
   production polish. Do not look up reviews or any automated rating of the episode. Each aspect
   is separate: a messy structure does not lower depth.
5. Unsure? Use *cannot determine*, *requires subject expertise* or *insufficient evidence*. An
   honest "can't tell" is more useful than a guess.
6. *Personal enjoyment* is optional and separate. It is never used as a quality rating.
7. Press **Mark this item complete** for each finished item, then **Download ratings file**, and
   send that file to the coordinator. You can send partial work and continue later.

## Consent (template, must be approved by the project owner before use)

The text that packets actually show is in
[`templates/rating-consent.txt`](templates/rating-consent.txt). The summary below is for
reference.

> I agree to rate the transcripts in this packet for the AI Video Discovery research project. I
> understand that:
>
> * my ratings, notes and citations are stored under a pseudonym, and my e-mail or name is not
>   stored with them;
> * my individual ratings, citations and notes are not published; only aggregates from at least
>   three raters may be (for example, agreement between raters);
> * publishing my individual ratings would need my separate, written agreement;
> * pseudonymous records can sometimes be re-identified (wording of notes, timing, few raters);
> * the coordinator knows who uses which pseudonym and keeps that private;
> * the reported time I spent on each item is recorded;
> * ratings files and the pseudonym list are deleted at the latest 12 months after my last
>   submission (period to be confirmed by the project owner);
> * I can withdraw at any time; my file is then deleted and excluded from later analyses, but
>   aggregates already published are not recalculated;
> * this is unpaid, voluntary work.

## What this does not provide

* Recruitment. The project has no budget and no rater pool.
* Verification of identity or expertise. Expertise tags are self-declared.
* Offline pairwise comparisons. Those still need the online annotation UI (`docs/annotation.md`).
* Protection against a rater who shares their packet. The secret authenticates the file, not
  the person, so treat it like a password.
