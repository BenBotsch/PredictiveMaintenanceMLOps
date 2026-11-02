"""Command line: ``pmm download | train | export | serve | drift``."""

import argparse
import json
from pathlib import Path

DATA = Path("data")
BUNDLE = Path("bundle")
TRACKING = "sqlite:///mlflow.db"
RESULTS = Path("results")


def main(argv: list[str] | None = None) -> None:
    """Run one command."""
    parser = argparse.ArgumentParser(prog="pmm", description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA, help="dataset directory")
    parser.add_argument("--tracking-uri", default=TRACKING, help="MLflow tracking and registry")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download", help="download AI4I 2020 and check its SHA-256")
    train_cmd = sub.add_parser("train", help="train the candidates, register and promote the best")
    train_cmd.add_argument("--seed", type=int, default=0)
    export_cmd = sub.add_parser("export", help="write the champion as a serving bundle")
    export_cmd.add_argument("--out", type=Path, default=BUNDLE)
    serve_cmd = sub.add_parser("serve", help="serve a bundle over HTTP")
    serve_cmd.add_argument("--bundle", type=Path, default=BUNDLE)
    serve_cmd.add_argument("--host", default="127.0.0.1")
    serve_cmd.add_argument("--port", type=int, default=8000)
    drift_cmd = sub.add_parser("drift", help="PSI of a CSV of readings against the training data")
    drift_cmd.add_argument("csv", type=Path)
    drift_cmd.add_argument("--bundle", type=Path, default=BUNDLE)
    args = parser.parse_args(argv)

    # Imports per command, so serving needs neither MLflow nor the training code.
    if args.command == "download":
        from pmm.data import download

        print(download(args.data))
    elif args.command in {"train", "export"}:
        from pmm import train
        from pmm.data import CSV_NAME, load, split

        # The split stays fixed, so validation scores of different seeds compare.
        splits = split(load(args.data / CSV_NAME))
        if args.command == "train":
            outcome = train.run(splits, args.tracking_uri, seed=args.seed)
            RESULTS.mkdir(exist_ok=True)
            result = {
                "validation": {c.name: c.val.to_dict() for c in outcome.candidates},
                "registered": {
                    "candidate": outcome.best.name,
                    "version": outcome.version,
                },
                "test": outcome.test.to_dict(),
            }
            (RESULTS / "training.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"test of {outcome.best.name}: {outcome.test.to_dict()}")
        else:
            print(train.export(args.tracking_uri, splits.train, args.out))
    elif args.command == "serve":
        import uvicorn

        from pmm import bundle
        from pmm.api import create_app

        uvicorn.run(create_app(bundle.load(args.bundle)), host=args.host, port=args.port)
    elif args.command == "drift":
        import pandas as pd

        from pmm import bundle, drift

        frame = pd.read_csv(args.csv)
        for item in drift.report(bundle.load(args.bundle).metadata.reference, frame):
            print(f"{item.feature:<22} PSI {item.psi:.4f}  {item.level}")
