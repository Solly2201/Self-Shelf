"""Self-Shelf command-line entry point.

Runs the full pricing pipeline: data preparation, demand model training and
evaluation, elasticity estimation, and economic price optimization.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from selfshelf.config import PricingConfig
from selfshelf.pipeline import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = REPO_ROOT / "data" / "walmart_large_sample_data_with_categories.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "final_optimized_prices.csv"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="self-shelf",
        description="Expiry-aware dynamic pricing engine",
    )
    parser.add_argument(
        "-n", "--items", type=int, default=25,
        help="number of test-set products to optimize (default: 25)",
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="path to the retail dataset CSV",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="path for the recommendations CSV",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="also write a per-product price sweep CSV next to the output",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="random seed for the whole run (default: from config)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.items <= 0:
        print("Number of items must be greater than 0.")
        return 1
    if not args.data.exists():
        print(f"Dataset not found: {args.data}")
        return 1

    config = PricingConfig()
    if args.seed is not None:
        config.seed = args.seed

    print("1. Preparing data and training the demand model...")
    result = run_pipeline(
        data_path=str(args.data),
        config=config,
        num_items=args.items,
        collect_sweeps=args.sweep,
        progress=True,
    )

    print("\n2. Demand model evaluation (daily demand, units):")
    for split_name, metrics in result.model_report.items():
        print(
            f"   {split_name:<11} MAE={metrics['mae']:.2f}  "
            f"RMSE={metrics['rmse']:.2f}  R2={metrics['r2']:.3f}"
        )

    print("\n3. Estimated department elasticities (from training data):")
    for dept, info in sorted(result.elasticities.items()):
        print(
            f"   {dept:<15} e={info['elasticity']:>6.2f}  "
            f"({info['source']}, n={info['n_observations']})"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.recommendations.to_csv(args.output, index=False)
    print(f"\n4. Wrote {len(result.recommendations)} recommendations to "
          f"{args.output}")

    if result.sweeps is not None:
        sweep_path = args.output.with_name(args.output.stem + "_sweep.csv")
        result.sweeps.to_csv(sweep_path, index=False)
        print(f"   Wrote price sweeps to {sweep_path}")

    actions = result.recommendations["Action"].value_counts().to_dict()
    print(f"\n5. Actions: {json.dumps(actions)}")
    print(f"   Run configuration: {json.dumps(result.config_summary)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
