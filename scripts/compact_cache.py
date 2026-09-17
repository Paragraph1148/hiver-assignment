"""Compact the committed LLM response cache.

The cache is append-only, so a re-run (or, as happened here, two concurrent
runs) writes the same key more than once. `_load` already takes the last entry,
so duplicates never affected correctness - but the file ships in the repo as the
reproduction mechanism, and it should not carry dead weight.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    path = Path("data/llm_cache.jsonl")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    keep: dict[str, dict] = {}
    for r in rows:                       # later wins, matching ResponseCache._load
        keep[r["key"]] = r
    before = path.stat().st_size
    with path.open("w") as fh:
        for k in sorted(keep):
            fh.write(json.dumps(keep[k], ensure_ascii=False) + "\n")
    after = path.stat().st_size
    print(f"{len(rows)} lines -> {len(keep)} unique keys "
          f"({before/1024:.0f} KB -> {after/1024:.0f} KB)")

    from hiver.llm.cache import ResponseCache
    assert len(ResponseCache(path, mode="replay")) == len(keep)
    print("reload check: cache still serves every key")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
