"""Show which LLM providers this machine can use, and prove replay needs none.

    python scripts/llm_doctor.py          # config only, no network
    python scripts/llm_doctor.py --probe  # send one tiny live request
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiver.llm.cache import DEFAULT_PATH, ResponseCache      # noqa: E402
from hiver.llm.client import LLM, _load_dotenv, detect_provider  # noqa: E402
from hiver.llm.providers import REGISTRY                     # noqa: E402
import os                                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="send one live request")
    args = ap.parse_args()

    _load_dotenv()
    print("Providers")
    for name in ("groq", "anthropic", "openai", "gemini"):
        cls = REGISTRY[name]
        ok = bool(os.environ.get(cls.env_key))
        print(f"  {'[x]' if ok else '[ ]'} {name:<10} {cls.env_key:<20} default={cls.default_model}")

    active = detect_provider()
    print(f"\nActive provider: {active or '(none — replay mode only)'}")
    if override := os.environ.get("LLM_MODEL"):
        print(f"Model override:  {override}")

    cache = ResponseCache(DEFAULT_PATH, mode="replay")
    print(f"Recorded cache:  {len(cache)} responses at {DEFAULT_PATH}")
    print("  -> `LLM_CACHE_MODE=replay` reproduces headline results with no key and no network.")

    if args.probe:
        if not active:
            print("\nNo key configured; nothing to probe.")
            return 1
        llm = LLM(mode="live")
        r = llm.ask("Reply with exactly: OK", max_tokens=500)
        print(f"\nProbe: {llm.provider_name}/{llm.model} -> {r.text.strip()[:40]!r} "
              f"({r.total_tokens} tok, {r.latency_ms:.0f}ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
