"""Build the SpotifyCares first-contact corpus from raw TWCS.

    python scripts/build_corpus.py --brand SpotifyCares
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiver.threads import RAW_DEFAULT, build_corpus, load_twcs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default="SpotifyCares")
    ap.add_argument("--raw", default=RAW_DEFAULT)
    ap.add_argument("--out", default="data/interim")
    args = ap.parse_args()

    print(f"loading {args.raw} ...", flush=True)
    df = load_twcs(args.raw)
    print(f"  {len(df):,} tweets", flush=True)

    print(f"reconstructing threads for {args.brand} ...", flush=True)
    c = build_corpus(df, args.brand)
    print(f"  {c}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    c.pairs.to_parquet(out / f"{args.brand}_pairs.parquet", index=False)
    c.threads.to_parquet(out / f"{args.brand}_threads.parquet", index=False)

    p = c.pairs
    print("\n--- corpus summary")
    print(f"  first-contact pairs : {len(p):,}")
    print(f"  date range          : {p['ts'].min()} .. {p['ts'].max()}")
    print(f"  median cust len     : {p['cust_len'].median():.0f} chars")
    print(f"  median reply len    : {p['reply_len'].median():.0f} chars")
    print(f"  deflection replies  : {p['is_deflection'].mean()*100:.1f}%")
    print(f"  threads with thanks : {p['got_thanks'].mean()*100:.1f}%")
    print(f"  signed by an agent  : {p['agent'].notna().mean()*100:.1f}%  "
          f"({p['agent'].nunique()} distinct)")
    print(f"  customer msgs w/ PII: {p['cust_has_pii'].mean()*100:.1f}%")
    print(f"  median thread turns : {p['n_turns'].median():.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
