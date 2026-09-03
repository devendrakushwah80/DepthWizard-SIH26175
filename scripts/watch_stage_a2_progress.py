"""Live terminal progress for the persistent Stage A2 Optuna study."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import optuna


ROOT = Path("outputs/player1_stage_a2/hpo")
STORAGE = f"sqlite:///{(ROOT / 'optuna_stage_a2.db').resolve().as_posix()}"
STUDY_NAME = "depthwizard_stage_a2"


def render() -> None:
    study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)
    trials = study.trials
    states = Counter(trial.state.name for trial in trials)
    latest = trials[-1] if trials else None
    completed = [trial for trial in trials if trial.state.name == "COMPLETE"]

    os.system("cls" if os.name == "nt" else "clear")
    print(f"DepthWizard Stage A2 HPO | {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 68)
    print(
        f"Total {len(trials):>2} | RUNNING {states['RUNNING']:>2} | "
        f"COMPLETE {states['COMPLETE']:>2} | PRUNED {states['PRUNED']:>2} | "
        f"FAILED {states['FAIL']:>2}"
    )

    if latest is not None:
        print(f"Current/latest trial: {latest.number} [{latest.state.name}]")
        if latest.intermediate_values:
            values = ", ".join(
                f"epoch {epoch}: {value:.6f}"
                for epoch, value in latest.intermediate_values.items()
            )
            print(f"Reported objectives: {values}")
        else:
            print("Reported objectives: waiting for the first epoch evaluation")

        history_path = ROOT / "trial_logs" / f"trial_{latest.number:03d}" / "history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text(encoding="utf-8"))
                if history:
                    row = history[-1]
                    metrics = row["metrics"]
                    print(
                        f"Last epoch {row['epoch']}/5 | train loss {row['train_loss']:.4f} | "
                        f"MAE {metrics['mae_m']:.4f} m | R2 {metrics['r2']:+.4f} | "
                        f"objective {row['objective']:.6f}"
                    )
            except (OSError, json.JSONDecodeError, KeyError):
                print("Epoch history is currently being updated...")

    if completed:
        best = min(completed, key=lambda trial: float(trial.value))
        print(f"Best completed trial: {best.number} | objective {best.value:.6f}")
    else:
        print("Best completed trial: none yet")
    print("\nRefreshes automatically. Press Ctrl+C to close this viewer only.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        while True:
            render()
            if args.once:
                return
            time.sleep(max(1.0, args.interval))
    except KeyboardInterrupt:
        print("\nProgress viewer closed; background training was not stopped.")


if __name__ == "__main__":
    main()
