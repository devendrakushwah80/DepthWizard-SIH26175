"""Stage A2 full-candidate training, full-tile selection, stability, and lock.

This worker never imports, reads, or evaluates the NYC manifest or dataset.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage_a2_development import dataset, ensure_baselines, make_sampler
from src.models.rdah_net import RDAHNetCore
from src.training.stage_a2_engine import set_determinism, train_model
from src.training.stage_a2_full_tile import evaluate_full_tile_checkpoint


ROOT = Path("outputs/player1_stage_a2")
FULL = ROOT / "full_candidates"
FINAL = ROOT / "final"
SELECTION_PATH = FULL / "candidate_selection.json"
PROGRESS_PATH = FULL / "progress_state.json"
WORKER_LOCK = FULL / "FULL_TRAINING_WORKER_LOCK.json"
MANIFESTS = Path("data/gamus/splits/stage_a2")
FINAL_MANIFEST = MANIFESTS / "final_dev_val.txt"
TRAIN_MANIFEST = MANIFESTS / "train_core.txt"
HPO_MANIFEST = MANIFESTS / "hpo_dev.txt"
M1_CHECKPOINT = Path("outputs/player1_stage_a/a1_rdah/checkpoints/best_mae_model.pth")
CONTROL_CHECKPOINT = ROOT / "m2_data_control/best_model.pth"
PRIMARY_SEED = 42
SECOND_SEED = 1337
MAX_EPOCHS = 15


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for attempt in range(40):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def update_progress(**updates) -> None:
    payload = load(PROGRESS_PATH) if PROGRESS_PATH.exists() else {}
    payload.update(updates)
    payload["updated_at"] = now_iso()
    if "phase_started_unix" in payload:
        payload["total_phase_elapsed_seconds"] = time.time() - payload["phase_started_unix"]
    dump(PROGRESS_PATH, payload)


def acquire_worker_lock(resume: bool = False) -> None:
    FULL.mkdir(parents=True, exist_ok=True)
    if (FINAL / "FINAL_MODEL_LOCK.json").exists():
        raise RuntimeError("FINAL_MODEL_LOCK.json already exists; refusing duplicate full-training run")
    payload = {
        "pid": os.getpid(),
        "started_at": now_iso(),
        "status": "RUNNING",
        "purpose": "full candidates -> FINAL-DEV-VAL -> stability -> M2-FINAL lock",
        "NYC_ALLOWED": False,
    }
    if resume:
        if not WORKER_LOCK.exists():
            raise RuntimeError("cannot resume: worker lock does not exist")
        previous = load(WORKER_LOCK)
        if previous.get("status") != "ERROR":
            raise RuntimeError(f"cannot resume worker with status {previous.get('status')}")
        previous.update(
            {
                "pid": os.getpid(),
                "status": "RUNNING",
                "resumed_at": now_iso(),
                "resume_count": int(previous.get("resume_count", 0)) + 1,
            }
        )
        dump(WORKER_LOCK, previous)
        return
    try:
        with WORKER_LOCK.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except FileExistsError as exc:
        raise RuntimeError(f"worker lock already exists: {WORKER_LOCK}") from exc


def exact_config(parameters: dict, seed: int) -> dict:
    config = dict(parameters)
    config.update(
        {
            "seed": seed,
            "epochs": MAX_EPOCHS,
            "weight_0_2": 1.0,
            "weight_2_10": 1.0,
            "optimizer": "AdamW",
            "crop_size": 512,
            "early_stopping_patience": 0,
            "loss_type": "height_aware",
        }
    )
    return config


def _archive_partial(out_dir: Path) -> None:
    if not out_dir.exists() or (out_dir / "training_summary.json").exists():
        return
    archive = out_dir.with_name(f"{out_dir.name}_interrupted_{datetime.now():%Y%m%d_%H%M%S}")
    if archive.exists():
        raise RuntimeError(f"partial archive already exists: {archive}")
    out_dir.replace(archive)


def train_candidate(candidate: dict, seed: int, out_dir: Path, candidate_index: int, stability: bool = False) -> dict:
    summary_path = out_dir / "training_summary.json"
    required = (
        summary_path,
        out_dir / "config.json",
        out_dir / "history.json",
        out_dir / "best_model.pth",
        out_dir / "latest_checkpoint.pth",
        out_dir / "best_metrics.json",
        out_dir / "sampler_stats.json",
        out_dir / "runtime_stats.json",
    )
    if all(path.exists() for path in required):
        summary = load(summary_path)
        if summary["epochs_completed"] != MAX_EPOCHS:
            raise RuntimeError(f"completed candidate has {summary['epochs_completed']} epochs, expected {MAX_EPOCHS}")
        return summary
    _archive_partial(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = exact_config(candidate["parameters"], seed)
    dump(
        out_dir / "config.json",
        {
            **config,
            "source_trial": candidate["trial_number"],
            "parameters_match_clean_optuna_trial_exactly": True,
            "metrics_during_training": "HPO-DEV PROXY METRICS",
        },
    )
    update_progress(
        status="RUNNING",
        phase="STABILITY" if stability else "TRAINING",
        candidate_trial=candidate["trial_number"],
        candidate_index=candidate_index,
        candidate_total=3,
        seed=seed,
        output_dir=str(out_dir),
        current_epoch=1,
        stability_status="TRAINING_SECOND_SEED" if stability else "NOT_STARTED",
        latest_non_finite_count=0,
    )
    train_ds = dataset("train_core.txt", True, config["augmentation_strength"])
    val_ds = dataset("hpo_dev.txt", False)
    hpo_baseline, hpo_tiles, _, _ = ensure_baselines()
    # Seed before model construction so every primary candidate starts from
    # the same deterministic parameter initialization.
    set_determinism(seed)
    model = RDAHNetCore(
        d_model=32,
        num_heads=4,
        output_parameterization=config["output_parameterization"],
    )
    summary = train_model(
        model,
        train_ds,
        make_sampler(train_ds, config),
        val_ds,
        config,
        out_dir,
        hpo_baseline,
        hpo_tiles,
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    del model, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def load_or_evaluate(
    spec: dict,
    model_index: int,
    model_total: int,
    phase: str = "FINAL-DEV-VAL",
) -> dict:
    out_dir = FULL / "final_dev_full_tile" / spec["key"]
    if (out_dir / "metrics.json").exists() and (out_dir / "per_tile_metrics.json").exists():
        return {"metrics": load(out_dir / "metrics.json"), "per_tile": load(out_dir / "per_tile_metrics.json")}

    def progress_callback(progress: dict) -> None:
        update_progress(
            phase=phase,
            evaluation_status="RUNNING",
            evaluation=progress,
            output_dir=str(out_dir),
        )

    update_progress(
        phase=phase,
        evaluation_status="RUNNING",
        evaluation={
            "model_name": spec["name"],
            "model_index": model_index,
            "model_total": model_total,
            "tile_number": 0,
            "tile_total": 200,
            "valid_tiles": 0,
            "running_mae_m": None,
        },
        output_dir=str(out_dir),
    )
    return evaluate_full_tile_checkpoint(
        model_name=spec["name"],
        checkpoint=Path(spec["checkpoint"]),
        output_parameterization=spec["output_parameterization"],
        manifest=FINAL_MANIFEST,
        out_dir=out_dir,
        progress_callback=progress_callback,
        model_index=model_index,
        model_total=model_total,
    )


def tile_comparison(model_rows: list[dict], comparator_rows: list[dict]) -> dict:
    model = {row["sample_id"]: row for row in model_rows}
    comparator = {row["sample_id"]: row for row in comparator_rows}
    ids = sorted(set(model) & set(comparator))
    if len(ids) != len(model) or len(ids) != len(comparator):
        raise RuntimeError("FINAL-DEV-VAL model tile sets do not match")
    delta = np.asarray([model[sid]["mae_m"] - comparator[sid]["mae_m"] for sid in ids])
    wins = int(np.sum(delta < -1e-9))
    ties = int(np.sum(np.abs(delta) <= 1e-9))
    return {
        "tile_count": len(ids),
        "wins": wins,
        "losses": len(ids) - wins - ties,
        "ties": ties,
        "win_rate": wins / len(ids),
        "mean_mae_delta_m": float(delta.mean()),
        "median_mae_delta_m": float(np.median(delta)),
    }


def _ordinal_ranks(values: dict[int, float], maximize: bool = False) -> dict[int, int]:
    ordered = sorted(values, key=lambda key: values[key], reverse=maximize)
    return {trial: rank for rank, trial in enumerate(ordered, 1)}


def compare_and_select(candidates: list[dict], results: dict[str, dict]) -> tuple[int, dict]:
    m1_rows = results["m1"]["per_tile"]
    control_rows = results["m2_data_control"]["per_tile"]
    comparisons = {}
    for key, result in results.items():
        comparisons[key] = {
            "vs_m1": tile_comparison(result["per_tile"], m1_rows),
            "vs_m2_data_control": tile_comparison(result["per_tile"], control_rows),
        }
        result["metrics"]["tile"]["vs_m1"] = comparisons[key]["vs_m1"]
        result["metrics"]["tile"]["vs_m2_data_control"] = comparisons[key]["vs_m2_data_control"]
        dump(FULL / "final_dev_full_tile" / key / "metrics.json", result["metrics"])

    candidate_metrics = {row["trial_number"]: results[f"trial_{row['trial_number']}"]["metrics"] for row in candidates}
    control = results["m2_data_control"]["metrics"]
    m1 = results["m1"]["metrics"]
    admissible = []
    for trial, metrics in candidate_metrics.items():
        low_ok = all(
            metrics["height_buckets"][bucket]["mae_m"]
            <= 1.07 * max(m1["height_buckets"][bucket]["mae_m"], control["height_buckets"][bucket]["mae_m"])
            for bucket in ("0-2m", "2-10m")
        )
        if low_ok:
            admissible.append(trial)
    selection_pool = admissible or list(candidate_metrics)
    criteria = [
        ("rmse", 10, False, lambda trial: candidate_metrics[trial]["rmse_m"]),
        ("mae", 9, False, lambda trial: candidate_metrics[trial]["mae_m"]),
        ("r2", 8, True, lambda trial: candidate_metrics[trial]["r2"]),
        ("building_mae", 7, False, lambda trial: candidate_metrics[trial]["building"]["mae_m"]),
        ("10_20_mae", 6, False, lambda trial: candidate_metrics[trial]["height_buckets"]["10-20m"]["mae_m"]),
        ("20_50_mae", 5, False, lambda trial: candidate_metrics[trial]["height_buckets"]["20-50m"]["mae_m"]),
        ("absolute_bias", 4, False, lambda trial: abs(candidate_metrics[trial]["bias_m"])),
        ("tile_win_vs_m1", 3, True, lambda trial: comparisons[f"trial_{trial}"]["vs_m1"]["win_rate"]),
        ("correlation", 2, True, lambda trial: candidate_metrics[trial]["pearson_r"] + candidate_metrics[trial]["spearman_rho"]),
        (
            "dominant_low_regimes",
            1,
            False,
            lambda trial: candidate_metrics[trial]["height_buckets"]["0-2m"]["mae_m"]
            + candidate_metrics[trial]["height_buckets"]["2-10m"]["mae_m"],
        ),
    ]
    score = {trial: 0 for trial in selection_pool}
    rank_details = {trial: {} for trial in selection_pool}
    for name, weight, maximize, getter in criteria:
        values = {trial: float(getter(trial)) for trial in selection_pool}
        ranks = _ordinal_ranks(values, maximize=maximize)
        for trial in selection_pool:
            score[trial] += weight * ranks[trial]
            rank_details[trial][name] = {"value": values[trial], "rank": ranks[trial], "weight": weight}
    winner = min(selection_pool, key=lambda trial: (score[trial], candidate_metrics[trial]["rmse_m"]))
    payload = {
        "status": "FINAL-DEV-VAL FULL-TILE selection complete",
        "created_at": now_iso(),
        "protocol": {
            "tile_size": 1024,
            "window": 512,
            "step": 256,
            "grid": "3x3",
            "blend": "Hanning weighted",
            "same_reconstruction_for_all_models": True,
        },
        "models": {key: value["metrics"] for key, value in results.items()},
        "tile_comparisons": comparisons,
        "selection": {
            "candidate_trials": list(candidate_metrics),
            "dominant_low_regime_admissible_trials": admissible,
            "selection_pool": selection_pool,
            "weighted_priority_rank_score_lower_is_better": score,
            "rank_details": rank_details,
            "selected_provisional_trial": winner,
            "gte_50m_used_as_selection_decider": False,
            "reason": "Selected by ordered multi-metric FINAL-DEV-VAL full-tile priorities after dominant-low-regime guard; >=50m was reported but excluded as a deciding criterion.",
        },
        "NYC_used": False,
    }
    dump(FULL / "final_dev_comparison.json", payload)
    make_comparison_markdown(payload)
    return winner, payload


def make_comparison_markdown(payload: dict) -> None:
    labels = ["m1", "m2_data_control", "trial_10", "trial_11", "trial_26"]
    lines = [
        "# Stage A2 FINAL-DEV-VAL Full-Tile Comparison",
        "",
        "All models use 1024x1024 reconstruction from nine 512x512 windows at step 256 with identical Hanning blending.",
        "",
        "| Model | MAE | RMSE | R2 | Pearson | Spearman | Bias | Building MAE | 10-20m | 20-50m | >=50m | Win vs M1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in labels:
        metrics = payload["models"][key]
        comparison = payload["tile_comparisons"][key]["vs_m1"]
        lines.append(
            f"| {key} | {metrics['mae_m']:.4f} | {metrics['rmse_m']:.4f} | {metrics['r2']:+.4f} | "
            f"{metrics['pearson_r']:+.4f} | {metrics['spearman_rho']:+.4f} | {metrics['bias_m']:+.4f} | "
            f"{metrics['building']['mae_m']:.4f} | {metrics['height_buckets']['10-20m']['mae_m']:.4f} | "
            f"{metrics['height_buckets']['20-50m']['mae_m']:.4f} | {metrics['height_buckets']['>=50m']['mae_m']:.4f} | "
            f"{comparison['wins']}/{comparison['tile_count']} ({comparison['win_rate']*100:.1f}%) |"
        )
    tall = payload["models"]["m1"]["height_buckets"][">=50m"]
    lines.extend(
        [
            "",
            f">=50m support: {tall['pixel_count']} pixels ({tall['support_percentage']:.6f}%), "
            f"across {tall['tile_count_with_support']} tiles. It was not used as the sole or deciding selection criterion.",
            "",
            f"Selected provisional trial: {payload['selection']['selected_provisional_trial']}.",
            "",
            "NYC has not been used or evaluated.",
        ]
    )
    (FULL / "final_dev_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def stability_analysis(trial: int, primary: dict, secondary: dict, comparison: dict, primary_comparison: dict) -> dict:
    fields = {
        "mae_m": (primary["mae_m"], secondary["mae_m"]),
        "rmse_m": (primary["rmse_m"], secondary["rmse_m"]),
        "r2": (primary["r2"], secondary["r2"]),
        "pearson_r": (primary["pearson_r"], secondary["pearson_r"]),
        "spearman_rho": (primary["spearman_rho"], secondary["spearman_rho"]),
        "bias_m": (primary["bias_m"], secondary["bias_m"]),
        "building_mae_m": (primary["building"]["mae_m"], secondary["building"]["mae_m"]),
        "10_20_mae_m": (primary["height_buckets"]["10-20m"]["mae_m"], secondary["height_buckets"]["10-20m"]["mae_m"]),
        "20_50_mae_m": (primary["height_buckets"]["20-50m"]["mae_m"], secondary["height_buckets"]["20-50m"]["mae_m"]),
        "gte_50_mae_m": (primary["height_buckets"][">=50m"]["mae_m"], secondary["height_buckets"][">=50m"]["mae_m"]),
        "full_tile_win_rate_vs_m1": (primary_comparison["win_rate"], comparison["win_rate"]),
    }
    deltas = {name: {"primary": a, "second_seed": b, "delta_second_minus_primary": b - a} for name, (a, b) in fields.items()}
    major_reasons = []
    thresholds = {
        "mae_m": 0.50,
        "rmse_m": 0.75,
        "r2": 0.08,
        "building_mae_m": 0.75,
        "10_20_mae_m": 1.25,
        "20_50_mae_m": 2.00,
        "full_tile_win_rate_vs_m1": 0.10,
    }
    for name, threshold in thresholds.items():
        if abs(deltas[name]["delta_second_minus_primary"]) > threshold:
            major_reasons.append(f"{name} delta exceeds {threshold}")
    payload = {
        "selected_trial": trial,
        "primary_seed": PRIMARY_SEED,
        "second_seed": SECOND_SEED,
        "deltas": deltas,
        "major_instability": bool(major_reasons),
        "major_instability_reasons": major_reasons,
        "automatic_retuning_performed": False,
        "NYC_used": False,
    }
    dump(FULL / "stability_analysis.json", payload)
    return payload


def create_final_lock(trial: int, candidate: dict, primary_summary: dict, metrics: dict, stability: dict) -> dict:
    FINAL.mkdir(parents=True, exist_ok=True)
    lock_path = FINAL / "FINAL_MODEL_LOCK.json"
    destination = FINAL / "M2_FINAL.pth"
    if lock_path.exists() or destination.exists():
        raise RuntimeError("final lock destination already exists; refusing overwrite")
    source = FULL / f"trial_{trial}" / "best_model.pth"
    shutil.copy2(source, destination)
    checkpoint_hash = sha256(destination)
    (FINAL / "checkpoint.sha256").write_text(f"{checkpoint_hash}  M2_FINAL.pth\n", encoding="utf-8")
    config = exact_config(candidate["parameters"], PRIMARY_SEED)
    audit = load(ROOT / "data_audit.json")
    cache_config_path = ROOT / "dav2_cache/cache_config.json"
    cache_manifest_path = ROOT / "dav2_cache/cache_manifest.csv"
    cache_config = load(cache_config_path)
    model = RDAHNetCore(d_model=32, num_heads=4, output_parameterization=config["output_parameterization"])
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    del model
    lock = {
        "model_name": "M2-FINAL",
        "selected_source_trial": trial,
        "selected_seed": PRIMARY_SEED,
        "selected_epoch": primary_summary["best_epoch"],
        "checkpoint_path": str(destination),
        "checkpoint": str(destination),
        "checkpoint_SHA256": checkpoint_hash,
        "checkpoint_sha256": checkpoint_hash,
        "architecture_name": "RDAHNetCore(d_model=32,num_heads=4)",
        "trainable_parameter_count": trainable_parameters,
        "DAV2_model_identity": cache_config.get("model_id"),
        "DAV2_cache_identity": {
            "config_path": str(cache_config_path),
            "config_sha256": sha256(cache_config_path),
            "manifest_path": str(cache_manifest_path),
            "manifest_sha256": sha256(cache_manifest_path),
            "sample_count": cache_config.get("sample_count"),
            "dtype": cache_config.get("cache_dtype"),
        },
        "exact_model_hyperparameters": {
            "d_model": 32,
            "num_heads": 4,
            "output_parameterization": config["output_parameterization"],
        },
        "exact_loss_hyperparameters": {
            key: config[key]
            for key in ("beta", "lambda_grad", "lambda_log", "weight_0_2", "weight_2_10", "weight_10_20", "weight_20_50", "weight_50_plus", "building_weight")
        },
        "exact_sampler_ratios": {
            key.removeprefix("sampler_"): config[key]
            for key in ("sampler_random", "sampler_building", "sampler_gt10", "sampler_gt20", "sampler_rare_tall")
        },
        "scheduler": config["scheduler"],
        "warmup_ratio": config["warmup_ratio"],
        "augmentation": config["augmentation_strength"],
        "output_parameterization": config["output_parameterization"],
        "hyperparameters": config,
        "train_manifest_path": str(TRAIN_MANIFEST),
        "train_manifest_SHA256": audit["manifest_sha256"]["train_core"],
        "HPO_dev_manifest_path": str(HPO_MANIFEST),
        "HPO_dev_manifest_SHA256": audit["manifest_sha256"]["hpo_dev"],
        "FINAL_DEV_VAL_manifest_path": str(FINAL_MANIFEST),
        "FINAL_DEV_VAL_manifest_SHA256": audit["manifest_sha256"]["final_dev_val"],
        "checkpoint_creation_timestamp": datetime.fromtimestamp(destination.stat().st_mtime).astimezone().isoformat(),
        "final_validation_metrics": metrics,
        "seed_stability_metrics": stability,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "CUDA_version": torch.version.cuda,
            "CUDA_available": torch.cuda.is_available(),
            "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "transformers": transformers.__version__,
            "optuna": optuna.__version__,
        },
        "selected_using_NYC": False,
        "tuned_using_NYC": False,
        "calibrated_using_NYC": False,
        "evaluated_on_NYC": False,
        "NYC_statement": "NYC has not been used or evaluated.",
    }
    dump(lock_path, lock)
    return lock


def validate_inputs() -> list[dict]:
    selection = load(SELECTION_PATH)
    candidates = selection["candidates"]
    numbers = [row["trial_number"] for row in candidates]
    if numbers != [10, 11, 26] or len(candidates) != 3:
        raise RuntimeError(f"candidate selection invariant failed: {numbers}")
    required = [TRAIN_MANIFEST, HPO_MANIFEST, FINAL_MANIFEST, M1_CHECKPOINT, CONTROL_CHECKPOINT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required non-NYC inputs: {missing}")
    audit = load(ROOT / "data_audit.json")
    for path, key in (
        (TRAIN_MANIFEST, "train_core"),
        (HPO_MANIFEST, "hpo_dev"),
        (FINAL_MANIFEST, "final_dev_val"),
    ):
        actual_hash = sha256(path)
        expected_hash = audit["manifest_sha256"][key]
        if actual_hash != expected_hash:
            raise RuntimeError(f"manifest hash changed for {path}: {actual_hash} != {expected_hash}")
    # Verify the clean DB remains exactly the frozen completed study.
    study = optuna.load_study(
        study_name="depthwizard_stage_a2",
        storage=f"sqlite:///{(ROOT / 'hpo/optuna_stage_a2.db').resolve().as_posix()}",
    )
    states = {name: sum(trial.state.name == name for trial in study.trials) for name in ("COMPLETE", "PRUNED", "FAIL", "RUNNING")}
    if len(study.trials) != 30 or states != {"COMPLETE": 20, "PRUNED": 10, "FAIL": 0, "RUNNING": 0}:
        raise RuntimeError(f"clean HPO changed after freeze: {states}")
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    candidates = validate_inputs()
    if args.validate_only:
        print(json.dumps({"status": "validated", "trials": [row["trial_number"] for row in candidates], "NYC_opened": False}, indent=2))
        return

    acquire_worker_lock(resume=args.resume)
    previous_progress = load(PROGRESS_PATH) if args.resume and PROGRESS_PATH.exists() else {}
    phase_started = previous_progress.get("phase_started_unix", time.time())
    update_progress(
        status="RUNNING",
        phase="TRAINING",
        phase_started_unix=phase_started,
        worker_pid=os.getpid(),
        candidate_total=3,
        completed_candidates=0,
        remaining_candidates=3,
        stability_status="NOT_STARTED",
        evaluation_status="NOT_STARTED",
        model_lock_status="NOT_STARTED",
        cuda_available=torch.cuda.is_available(),
        NYC_evaluated=False,
    )
    try:
        primary_summaries = {}
        for index, candidate in enumerate(candidates, 1):
            out_dir = FULL / f"trial_{candidate['trial_number']}"
            primary_summaries[candidate["trial_number"]] = train_candidate(candidate, PRIMARY_SEED, out_dir, index)
            update_progress(
                completed_candidates=index,
                remaining_candidates=3 - index,
                last_completed_trial=candidate["trial_number"],
            )

        control_summary = load(ROOT / "m2_data_control/training_summary.json")
        specs = [
            {
                "key": f"trial_{candidate['trial_number']}",
                "name": f"Trial {candidate['trial_number']}",
                "checkpoint": str(FULL / f"trial_{candidate['trial_number']}/best_model.pth"),
                "output_parameterization": candidate["parameters"]["output_parameterization"],
            }
            for candidate in candidates
        ]
        specs.extend(
            [
                {
                    "key": "m2_data_control",
                    "name": "M2-DATA-CONTROL",
                    "checkpoint": str(CONTROL_CHECKPOINT),
                    "output_parameterization": control_summary["config"]["output_parameterization"],
                },
                {
                    "key": "m1",
                    "name": "M1",
                    "checkpoint": str(M1_CHECKPOINT),
                    "output_parameterization": "unconstrained",
                },
            ]
        )
        results = {spec["key"]: load_or_evaluate(spec, index, len(specs)) for index, spec in enumerate(specs, 1)}
        update_progress(evaluation_status="COMPLETE")
        winner, comparison_payload = compare_and_select(candidates, results)
        winner_candidate = next(row for row in candidates if row["trial_number"] == winner)
        update_progress(
            phase="STABILITY",
            selected_provisional_trial=winner,
            stability_status="TRAINING_SECOND_SEED",
        )
        second_dir = FULL / f"trial_{winner}_seed_{SECOND_SEED}"
        train_candidate(winner_candidate, SECOND_SEED, second_dir, next(index for index, row in enumerate(candidates, 1) if row["trial_number"] == winner), stability=True)
        second_spec = {
            "key": f"trial_{winner}_seed_{SECOND_SEED}",
            "name": f"Trial {winner} seed {SECOND_SEED}",
            "checkpoint": str(second_dir / "best_model.pth"),
            "output_parameterization": winner_candidate["parameters"]["output_parameterization"],
        }
        secondary = load_or_evaluate(second_spec, 1, 1, phase="STABILITY")
        secondary_vs_m1 = tile_comparison(secondary["per_tile"], results["m1"]["per_tile"])
        primary_key = f"trial_{winner}"
        primary_vs_m1 = comparison_payload["tile_comparisons"][primary_key]["vs_m1"]
        stability = stability_analysis(
            winner,
            results[primary_key]["metrics"],
            secondary["metrics"],
            secondary_vs_m1,
            primary_vs_m1,
        )
        update_progress(phase="MODEL LOCK", stability_status="COMPLETE", model_lock_status="RUNNING")
        lock = create_final_lock(
            winner,
            winner_candidate,
            primary_summaries[winner],
            results[primary_key]["metrics"],
            stability,
        )
        update_progress(
            status="COMPLETE",
            phase="MODEL LOCK",
            model_lock_status="COMPLETE",
            selected_final_trial=winner,
            selected_seed=PRIMARY_SEED,
            selected_epoch=primary_summaries[winner]["best_epoch"],
            final_checkpoint=lock["checkpoint_path"],
            final_checkpoint_sha256=lock["checkpoint_SHA256"],
            NYC_evaluated=False,
        )
        worker = load(WORKER_LOCK)
        worker.update({"status": "COMPLETE", "completed_at": now_iso(), "final_trial": winner})
        dump(WORKER_LOCK, worker)
        print(json.dumps({"status": "COMPLETE", "selected_trial": winner, "lock": str(FINAL / 'FINAL_MODEL_LOCK.json'), "NYC_evaluated": False}, indent=2))
    except Exception as exc:
        update_progress(status="ERROR", error=repr(exc), traceback=traceback.format_exc(), NYC_evaluated=False)
        worker = load(WORKER_LOCK)
        worker.update({"status": "ERROR", "failed_at": now_iso(), "error": repr(exc)})
        dump(WORKER_LOCK, worker)
        raise


if __name__ == "__main__":
    main()
