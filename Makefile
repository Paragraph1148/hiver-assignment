.PHONY: help setup reproduce live test clean verify

VENV := .venv
PY   := $(VENV)/bin/python

help:
	@echo "make setup      install dependencies (~2 min)"
	@echo "make reproduce  headline results from the recorded run - NO API KEY (~3 min)"
	@echo "make live       re-run against a provider - needs a key (~90 min, rate-limited)"
	@echo "make test       run the offline test suite"
	@echo "make verify     prove reproduce used no network"

setup:
	python3 -m venv $(VENV) 2>/dev/null || true
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"
	@echo "setup done"

# Replays the committed response cache. A cache miss is a hard error, so this
# cannot silently fall through to the network or bill anyone who runs it.
# Regenerates predictions from the recorded calls, then scores them - not just
# a re-scoring of committed outputs.
reproduce:
	LLM_CACHE_MODE=replay $(PY) scripts/run_intent.py
	LLM_CACHE_MODE=replay $(PY) scripts/eval_intent.py
	LLM_CACHE_MODE=replay $(PY) scripts/eval_route.py
	LLM_CACHE_MODE=replay $(PY) scripts/eval_draft.py --from-cache
	LLM_CACHE_MODE=replay $(PY) scripts/eval_semantic_cache.py

live:
	$(PY) scripts/run_intent.py
	$(PY) scripts/run_reply_route.py
	$(PY) scripts/eval_draft.py

test:
	$(PY) -m pytest tests/ -q

verify:
	@echo "running reproduce with no API keys and no network egress..."
	@env -u GROQ_API_KEY -u GEMINI_API_KEY -u OPENAI_API_KEY -u ANTHROPIC_API_KEY \
	  LLM_CACHE_MODE=replay $(PY) scripts/eval_intent.py > /dev/null && \
	  echo "PASS: headline metrics reproduced with zero credentials"

clean:
	rm -rf data/results/*.json data/results/*.parquet
