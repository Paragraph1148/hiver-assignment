# A support agent for @SpotifyCares, and the evidence it works

Classifies an incoming customer tweet, drafts a reply grounded in how Spotify's
agents actually handled similar cases, and decides whether to auto-send or
escalate — with a stated reason.

The system is ordinary. **The evaluation is the deliverable**, so most of what
follows is about how far the numbers can be trusted, and where they cannot.

---

## Reproduce the headline numbers — no API key, ~3 minutes

```bash
make setup
make reproduce
```

Every LLM call made during evaluation is recorded in `data/llm_cache.jsonl`,
keyed by `sha256(provider, model, messages, params)`. `make reproduce` replays
them, so you regenerate the predictions and get identical numbers with **no
credentials and no network**. Verified:

```
$ make verify
249 cache hits, 0 misses, 0 failed calls    # 76 seconds, all keys unset
PASS: headline metrics reproduced with zero credentials
```

A cache miss in replay mode is a hard error, never a silent fall-through to the
network — a cache that quietly calls out turns "reproducible" into "reproducible
if you happen to hold a key", and bills whoever runs it. `make live` re-runs
against a real provider (~90 minutes, rate-limited).

This recorded cache is **not** an inference feature and never runs in the live
agent. A separate semantic cache (`src/hiver/semantic_cache.py`) is the
production cost lever, and it is measured rather than assumed — see below.

**Bring your own model.** Groq, OpenAI, Anthropic and Gemini are all supported
behind one interface; whichever key you export is auto-detected.

```bash
export GROQ_API_KEY=...        # or ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
python scripts/llm_doctor.py   # shows which lanes are live
```

---

## Choosing the brand was the first real decision

Most brands in this dataset cannot support this task at all. A public reply can
look like support and carry none: it deflects to DM, punts to a URL, or is a
bare apology. Measured across every brand with ≥5,000 replies:

| brand | replies | deflect% | linkpunt% | thin% | diagnostic% | **thanked%** | groundable |
|---|---:|---:|---:|---:|---:|---:|---:|
| AmazonHelp | 168,814 | 0.7 | 3.2 | 6.1 | 4.6 | 4.5 | 152,376 |
| **AppleSupport** | 106,646 | **52.6** | 5.4 | 1.3 | 17.0 | 5.0 | 48,062 |
| **SpotifyCares** | 43,092 | 30.9 | 2.1 | 0.9 | 12.0 | **7.8** | 28,740 |
| Uber_Support | 56,160 | 35.9 | 13.2 | 4.2 | 1.2 | 5.2 | 28,466 |
| sprintcare | 22,209 | 48.0 | 1.4 | 7.0 | 3.7 | 2.9 | 10,025 |

**AppleSupport is the trap** — the obvious pick on volume, but 52.6% of its
public replies are "please DM us", so more than half the corpus teaches an agent
to deflect. **AmazonHelp is the second trap**: 90% groundable but only 4.6%
diagnostic and 4.5% thanked, because its public support is a routing desk, not a
resolution desk — the quality ceiling on a drafted reply is "apologise and
route", which makes the auto-handle decision degenerate.

SpotifyCares wins on the metrics that matter rather than the one that doesn't:
the highest resolution signal of any high-volume brand, real diagnostic
behaviour, and 28.7k groundable examples. It also signs replies — **208 distinct
human agents** — which makes an important limitation measurable.

Reproduce with `python scripts/brand_survey.py`.

---

## Results: intent classification

n=249, 10 classes, 95% CIs from 2,000 bootstrap resamples.

| model | accuracy [95% CI] | macro-F1 |
|---|---|---:|
| trivial: majority class | 20.1% [15.3–24.9] | 3.3% |
| simple: TF-IDF + logistic regression | 51.8% [45.4–58.2] | 50.2% |
| simple: hybrid-retrieval kNN | 54.6% [48.6–60.6] | 51.2% |
| **system: retrieval-augmented LLM** | **69.1% [63.5–74.7]** | 64.4% |
| *annotator self-consistency* | *71.9%* | — |

Every gap is significant under a paired bootstrap (p<0.001).

### What is misleading about that headline number

**69.1% is not a quality figure. The confidence interval contains the
annotator's own self-consistency of 71.9%.** The system is statistically
indistinguishable from the human who wrote the labels, so this test set can no
longer separate them, and further tuning against it fits annotator noise rather
than learning the task.

That ceiling is measured, not assumed: 32 golden items were re-labelled blind a
week apart. The annotator agreed with themselves 65.6% of the time on intent
(71.9% once a mid-project schema change is excluded), against **96.9% on the
auto-vs-escalate decision, Cohen's κ 0.871**. Intent labels are noisy; the
routing label — the one that actually matters operationally — is solid.

Full method and limitations: [`docs/annotation.md`](docs/annotation.md).

---

## The semantic cache, measured

A cosine-nearest response cache is a real cost lever on a repetitive inbox, but
the size of the win has to be measured. Hit rate on 10,000 messages in arrival
order; error rate is the share of served answers written for a different intent:

| cosine | calls saved | wrong intent served |
|---:|---:|---:|
| 0.95 | 1.5% | 0.0% |
| 0.90 | 4.6% | 1.1% |
| 0.85 | 12.4% | 2.6% |
| 0.80 | 25.3% | 6.6% |

At the intuitively safe 0.95 threshold it saves almost nothing. It only becomes
a real lever below 0.85, where you are knowingly serving some customers an
answer written for a different problem. A decision from a curve, not a free win.

---

## What I chose not to build

- **No fine-tuning.** No budget, and a tuned classifier would have been a weaker
  demonstration than the measurement work, given the label ceiling above.
- **No multi-turn dialogue.** First-contact triage only — one inbound message,
  one decision.
- **No live Twitter integration.** Out of scope for evaluating the agent.
- **No multilingual handling.** The corpus has non-English messages; they are
  labelled and scored like any other, and the failure analysis says what that
  costs.

---

## Layout

```
src/hiver/
  llm/           provider-agnostic client + recorded-response cache
  threads.py     TWCS thread reconstruction
  pii.py         scrubbing (typed placeholders, not deletion)
  embed.py       local MiniLM embeddings, cached
  cluster.py     UMAP + HDBSCAN intent discovery
  retrieve.py    hybrid BM25 + dense index
  classify.py    intent: 2 baselines + retrieval kNN + LLM
  draft.py       replies: canned + nearest-reply + RAG
  route.py       escalation: always-auto + intent prior + LLM policy
  judge.py       LLM-as-judge, cross-family, position-swapped
  metrics.py     bootstrap CIs, paired bootstrap, ECE, deferral curves
  semantic_cache.py   production response cache (distinct from the eval cache)
tools/           the labelling console used to build the golden set
docs/            annotation method, report
```

`make test` runs the offline suite (no network, no keys).
