# The golden set: how it was sampled and labelled

250 items, 249 usable. Single annotator, two passes, ~2.5 hours total.

## Sampling

Drawn from 26,095 first-contact pairs (a customer's opening tweet to
@SpotifyCares plus the brand's first public reply), stratified by **intent x
region** across 16 cells with a floor of 8 per cell, seed `20260910`.

Stratified rather than uniform because a uniform draw leaves rare intents with
~5 examples, where per-class F1 is noise. Every row carries a `weight` — how
many corpus messages it stands for — so metrics are reported both raw and
re-weighted to true frequency. On escalation the two agree closely (19.7% raw vs
19.2% weighted), so stratification did not distort the headline.

`region` is a stratum in its own right: `core` (densely clustered), `tail`
(atypical phrasing of a known intent), `other` (irreducibly heterogeneous). The
prediction that the tail is harder is only testable if the sample deliberately
contains enough of it.

The 250 threads are recorded in `held_out_threads.json` and excluded from the
retrieval index. Without that the evaluation retrieves its own answers.

## What the annotator could see

Only the customer's message. Withheld: the weak cluster label, the region tag,
and **Spotify's actual reply**. Any of them on screen would anchor the label, and
the set would inherit the clusterer's mistakes rather than independently check
them. The build asserts this on the parsed page payload.

There was **no LLM pre-fill**. An annotator who accepts a proposal has not
independently labelled anything, and model-human agreement measured that way is
an artefact of the process. The model is run separately over the same items.

## Conventions the annotator adopted

- **Messages whose content is an image or link** — labelled
  `non_actionable_generic` + `triage_question`. TWCS carries the tweet text but
  not attached media, so the actual complaint ("what's going on <URL>") is simply
  not in the dataset. 2.8% of the golden set and 1.9% of the full corpus. These
  are information-theoretically unanswerable from text, and asking for detail is
  the correct action, so the label is right and the item is not an error case.
- **"Wants a human"** — `asks_for_human`, added in pass 2 (see below).
- **Escalate means**: this cannot be safely auto-sent, judged from the customer's
  message alone — the information a live agent has before replying.

## Two passes, and why

Pass 1 covered all 250 over three sessions across a week. It left 24 items
incomplete, 20 of which carried a note: the annotator was stuck, not skipping.
The notes showed the schema, not the annotator, was at fault:

- **24 notes said "feature request".** There was no such intent. The clustering
  had proposed `platform_feature_requests` at ~9.9% before an LLM merge pass
  dissolved it into playback and playlist. Hand-labelling recovered a real class
  that automated taxonomy induction had discarded. It is 10.0% of the final set.
- **8 notes said the customer wanted a person.** With no category for it the
  annotator used `angry_or_vulnerable`, overloading it with messages carrying no
  anger. `asks_for_human` was added; 6 of those 7 items moved to it on review.

Pass 2 re-labelled 83 items — 24 incomplete, 20 schema-gap, 7 overloaded, and 32
pre-marked for the agreement check — presented **blind**, with no prior label
shown. The annotator's own note was shown, being their reasoning rather than an
answer. 19 of the 20 schema-gap items became `feature_request`, confirming the
diagnosis.

Labels were **not** revised by the system's author. A golden set edited by
whoever built the model under test cannot measure that model.

## Agreement and drift

32 items were re-labelled blind about a week after first being seen.

| | raw agreement | Cohen's kappa |
|---|---:|---:|
| intent | 65.6% | 0.589 |
| intent, excluding the schema change | **71.9%** | — |
| route (auto vs escalate) | **96.9%** | 0.871 |
| escalation reason | 84.4% | — |

Only 2 of 11 intent disagreements were caused by `feature_request` becoming
available, so 71.9% is the honest self-consistency figure.

**This is the most important number in the evaluation.** The annotator agrees
with themselves on intent about 72% of the time, so an intent classifier scoring
much above that is fitting annotator noise, not learning the task. Routing is a
different story: 96.9% agreement (kappa 0.871) means the auto-vs-escalate
decision — the one that actually matters operationally — rests on a far more
stable label than the intent taxonomy does.

Disagreements concentrate on two boundaries: `other` vs `non_actionable_generic`
(3) and `playback_and_app_issues` vs everything (5). Those are taxonomy design
problems, not carelessness.

Escalation rate by session ran 19.7% -> 21.7% -> 16.4%, and
`angry_or_vulnerable` use ran 4.4% -> 0% -> 1.5% as the annotator's mental model
settled. The drift is mild and is reported rather than smoothed away.

## Known limitations

1. **Single annotator.** Inter-annotator agreement is unmeasured; self-consistency
   is a weaker substitute and is generally an optimistic bound on it.
2. **The annotator also commissioned the system**, so the labels are not
   adversarial to it.
3. **Escalation has no ground truth.** These are one person's judgements about
   what a human should handle, not observed outcomes. They are validated against
   observable proxies (thread length, DM handoff, expressed dissatisfaction)
   rather than treated as truth.
4. **2.8% of items are unanswerable from text at all** (image-only content),
   which puts a hard ceiling on any reply-quality metric computed over them.
