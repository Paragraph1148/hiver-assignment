# Labelling console

`label_ui.html` is the instrument used to build the golden set. It is committed
so the labelling process is reproducible, not just its output.

Regenerate after re-sampling:

    python scripts/sample_golden.py      # draws the stratified sample
    python scripts/build_label_ui.py     # embeds it into the console

Two builds come out of one template: `label_ui.html` opens straight from disk,
and `label_ui.artifact.html` is the hosted version (it persists labels
server-side so they can be read back without moving files around).

## What the annotator can and cannot see

The page is given **only** an item id and the customer's message. The weak
cluster label, the region tag, and the brand's actual reply are all withheld.
Any of them on screen would anchor the annotator and the golden set would
inherit the clusterer's mistakes rather than provide an independent check on
them. `build_label_ui.py` asserts this rather than trusting it: the build fails
if an embedded item carries any field beyond `id` and `text`.

For the same reason there is no LLM pre-fill. The model is run over the same
items separately, so model-vs-human agreement is a measured result instead of a
number contaminated by the human having seen the model's answer first.

## Keys

    1-9              intent
    A S D            auto-handle, with reason
    Z X C V B N M    escalate, with reason
    F                flag as hard
    /                note
    left arrow       previous item

Two keystrokes finish an item. The physical row encodes the routing decision:
home row auto-handles, bottom row escalates.

Progress is written to `localStorage` on every keystroke and mirrored to the
artifact's document store when that is available, so a closed tab loses nothing
and the session resumes at the first unlabelled item.
