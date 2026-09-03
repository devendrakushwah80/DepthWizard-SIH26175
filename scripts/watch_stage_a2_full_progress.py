"""Read-only live viewer for the detached Stage A2 full-training worker."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("outputs/player1_stage_a2")
FULL = ROOT / "full_candidates"
STATE_PATH = FULL / "progress_state.json"
LOCK_PATH = ROOT / "final/FINAL_MODEL_LOCK.json"


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def metric(metrics: dict, key: str, default=0.0):
    value = metrics.get(key, default)
    return default if value is None else value


def render() -> None:
    state = read_json(STATE_PATH, {})
    os.system("cls" if os.name == "nt" else "clear")
    seed_stability_run = state.get("workflow") == "THREE_SEED_STABILITY"
    nyc_final_run = state.get("workflow") == "NYC_FINAL_UNSEEN"
    print("=" * 68)
    print(
        "DepthWizard - FINAL NYC UNSEEN EVALUATION"
        if nyc_final_run
        else (
            "DepthWizard Stage A2 - SEED STABILITY RUN"
            if seed_stability_run
            else "DepthWizard Stage A2 FULL TRAINING"
        )
    )
    print(f"Timestamp: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("=" * 68)
    if not state:
        print("Waiting for background worker state...")
        return
    phase = state.get("phase", "UNKNOWN")
    stability_evaluating = phase == "STABILITY" and state.get("evaluation_status") == "RUNNING"
    print(f"\nPHASE: {phase} | STATUS: {state.get('status', 'UNKNOWN')}")
    print(f"Worker PID: {state.get('worker_pid', 'unknown')}")
    if nyc_final_run:
        print(f"Model: {state.get('model', 'M2-FINAL')}")
        print(
            f"Trial: {state.get('trial', 10)} | Seed: {state.get('seed', 1337)} | "
            f"Epoch: {state.get('epoch', 10)}"
        )
        print(f"Checkpoint SHA verified: {state.get('checkpoint_SHA_verified', False)}")
        print(f"CUDA available: {state.get('cuda_available', False)}")

    training_phase = phase in {"TRAINING", "STABILITY", "SEED_STABILITY_TRAINING"}
    if training_phase and not stability_evaluating:
        trial = state.get("candidate_trial")
        if seed_stability_run:
            print(f"Trial: {trial}")
        else:
            print(f"Candidate: Trial {trial} | Candidate {state.get('candidate_index', '?')}/3")
        print(f"Seed: {state.get('seed', '?')}")
        run_dir = Path(state.get("output_dir", ""))
        history = read_json(run_dir / "history.json", []) if run_dir else []
        summary = read_json(run_dir / "training_summary.json", {}) if run_dir else {}
        if history:
            latest = history[-1]
            current_epoch = len(history) if summary else min(15, len(history) + 1)
            best = min(history, key=lambda row: row["objective"])
            metrics = latest["metrics"]
            buckets = metrics["height_buckets"]
            print(f"Epoch: {current_epoch}/15 | latest evaluated: {latest['epoch']}/15")
            print("\nLatest epoch:")
            print(
                f"  train loss {latest['train_loss']:.4f} | balanced objective {latest['objective']:.6f} | "
                f"MAE {metrics['mae_m']:.4f} | RMSE {metrics['rmse_m']:.4f}"
            )
            print(
                f"  R2 {metrics['r2']:+.4f} | Pearson {metrics['pearson_r']:+.4f} | "
                f"Spearman {metrics['spearman_rho']:+.4f} | Bias {metrics['bias_m']:+.4f}"
            )
            print(
                f"  Building MAE {metrics['building']['mae_m']:.4f} | 10-20m {buckets['10-20m']['mae_m']:.4f} | "
                f"20-50m {buckets['20-50m']['mae_m']:.4f} | >=50m {buckets['>=50m']['mae_m']:.4f}"
            )
            print(
                f"  +/-1m {metrics['within_1m_pct']:.2f}% | +/-2m {metrics['within_2m_pct']:.2f}% | "
                f"+/-5m {metrics['within_5m_pct']:.2f}% | +/-10m {metrics['within_10m_pct']:.2f}%"
            )
            best_metrics = best["metrics"]
            print("\nBest so far:")
            print(
                f"  epoch {best['epoch']} | objective {best['objective']:.6f} | MAE {best_metrics['mae_m']:.4f} | "
                f"RMSE {best_metrics['rmse_m']:.4f} | R2 {best_metrics['r2']:+.4f}"
            )
            total_elapsed = time.time() - state.get("phase_started_unix", time.time())
            print("\nRuntime:")
            print(
                f"  epoch {latest['epoch_seconds']:.1f}s | candidate {latest.get('candidate_elapsed_seconds', 0):.1f}s | "
                f"total phase {total_elapsed:.1f}s"
            )
            print("\nNumerics:")
            print(
                f"  AMP skipped {latest.get('amp_skipped_steps_total', 0)} | "
                f"non-finite {latest.get('non_finite_count', 0)} | CUDA {state.get('cuda_available')} | "
                f"peak VRAM {latest.get('peak_vram_mb', 0):.1f} MB"
            )
        else:
            print("Epoch: 1/15")
            print("\nWaiting for first epoch evaluation")

    seed_stability_evaluating = phase == "SEED_STABILITY_FINAL_DEV_VAL"
    nyc_evaluating = phase == "FINAL_NYC_UNSEEN_EVALUATION"
    if phase == "FINAL-DEV-VAL" or stability_evaluating or seed_stability_evaluating or nyc_evaluating:
        evaluation = state.get("evaluation", {})
        if seed_stability_evaluating:
            print("Evaluation phase: FINAL-DEV-VAL full-tile")
            print(f"Current model: {evaluation.get('model_name', 'Trial 10 seed 2026')}")
        print(f"Evaluating: {evaluation.get('model_name', 'waiting')}")
        if nyc_evaluating:
            print(
                f"Tile {evaluation.get('tile_number', 0)}/{evaluation.get('tile_total', 500)} | "
                f"valid evaluated {evaluation.get('valid_tiles', 0)}"
            )
        else:
            print(
                f"Model {evaluation.get('model_index', 0)}/{evaluation.get('model_total', 0)} | "
                f"tile {evaluation.get('tile_number', 0)}/{evaluation.get('tile_total', 200)} | "
                f"valid {evaluation.get('valid_tiles', 0)}"
            )
        running = evaluation.get("running_mae_m")
        print(f"Running MAE: {running:.4f} m" if running is not None else "Running MAE: waiting")
        if nyc_evaluating:
            print(f"Skipped tiles: {evaluation.get('skipped_tiles', 0)}")
            print(f"Elapsed runtime: {evaluation.get('elapsed_seconds', 0):.1f}s")

    lock = read_json(LOCK_PATH, {})
    print("\nProgress:")
    if nyc_final_run:
        evaluation = state.get("evaluation", {})
        print(
            f"  NYC tiles: {evaluation.get('tile_number', 0)}/{evaluation.get('tile_total', state.get('manifest_tiles', 500))} | "
            f"valid {evaluation.get('valid_tiles', state.get('evaluated_tiles', 0))} | "
            f"skipped {evaluation.get('skipped_tiles', state.get('skipped_tiles', 0))}"
        )
        print(f"  checkpoint SHA verified: {state.get('checkpoint_SHA_verified', False)}")
        print(f"  post-NYC tuning: {state.get('post_NYC_tuning_performed', False)}")
    elif seed_stability_run:
        print(f"  seed 2026 training: {'COMPLETE' if phase != 'SEED_STABILITY_TRAINING' else 'RUNNING'}")
        print(f"  FINAL-DEV-VAL: {state.get('evaluation_status', 'NOT_STARTED')}")
        print(f"  three-seed analysis: {state.get('three_seed_analysis_status', 'NOT_STARTED')}")
        print(f"  model lock: {state.get('model_lock_status', 'PRESERVED_PENDING_STABILITY')}")
        if state.get("stability_classification"):
            print(f"  classification: {state['stability_classification']}")
            print(f"  representative seed: {state.get('representative_seed')}")
    else:
        print(
            f"  completed candidates {state.get('completed_candidates', 0)}/3 | "
            f"remaining {state.get('remaining_candidates', 3)}"
        )
        print(f"  second-seed stability: {state.get('stability_status', 'NOT_STARTED')}")
        print(f"  FINAL-DEV-VAL: {state.get('evaluation_status', 'NOT_STARTED')}")
        print(f"  model lock: {state.get('model_lock_status', 'NOT_STARTED')}")
    show_lock = lock and (
        (not seed_stability_run and not nyc_final_run)
        or phase in {"THREE_SEED_COMPLETE", "FINAL_NYC_UNSEEN_COMPLETE"}
    )
    if show_lock:
        print("\nMODEL LOCK COMPLETE")
        print(
            f"  Trial {lock['selected_source_trial']} | seed {lock['selected_seed']} | "
            f"epoch {lock['selected_epoch']}"
        )
        print(f"  checkpoint: {lock['checkpoint_path']}")
        print(f"  SHA256: {lock['checkpoint_SHA256']}")
        print(f"  NYC evaluated = {str(lock['evaluated_on_NYC']).lower()}")
    if state.get("status") == "ERROR":
        print(f"\nERROR: {state.get('error')}")
    if nyc_final_run:
        print("\nCtrl+C closes this viewer only; it does not stop the background evaluation.")
    else:
        print("\nCtrl+C closes this viewer only; it does not stop the background training.")


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
        print("\nViewer closed. Background training continues unchanged.")


if __name__ == "__main__":
    main()
