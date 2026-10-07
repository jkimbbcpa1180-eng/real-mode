"""Run a reproducible synthetic demo without creating a persistent ledger."""
import argparse
import json
from dataclasses import asdict
from .master import RealModeMaster, URK2Input


def main():
    parser = argparse.ArgumentParser(description="Repaired Real Mode legacy demo")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.demo:
        parser.print_help()
        return
    master = RealModeMaster(seed=args.seed)
    result = master.execute_full_cycle(URK2Input(.85, .20, .55, .95, .30, .75))
    payload = {"kind": "synthetic_demo", "score_status": "uncalibrated_heuristic",
               "seed": args.seed, "result": asdict(result)}
    print(json.dumps(payload, indent=2, allow_nan=False) if args.json else
          "🚂 Choo choo — synthetic legacy demo\n" + json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
