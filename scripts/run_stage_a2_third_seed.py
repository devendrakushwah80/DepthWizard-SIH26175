"""Resolve Trial 10 seed stability with one deterministic third seed.

The worker has no NYC path or dataset dependency.  It trains seed 2026 once,
evaluates the frozen FINAL-DEV-VAL manifest, applies a predeclared robust
three-seed rule, and updates the model lock only when the rule finds a
defensible representative checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_stage_a2_development import dataset, make_sampler
from src.models.rdah_net import RDAHNetCore
from src.training.stage_a2_engine import set_determinism, train_model
from src.training.stage_a2_full_tile import evaluate_full_tile_checkpoint


ROOT = Path("outputs/player1_stage_a2")
FULL = ROOT / "full_candidates"
FINAL = ROOT / "final"
PROGRESS_PATH = FULL / "progress_state.json"
WORKER_LOCK = FULL / "THREE_SEED_WORKER_LOCK.json"
TRAIN_DIR = FULL / "trial_10_seed_2026"
EVAL_DIR = FULL / "final_dev_full_tile/trial_10_seed_2026"
MANIFESTS = Path("data/gamus/splits/stage_a2")
FINAL_MANIFEST = MANIFESTS / "final_dev_val.txt"
TRAIN_MANIFEST = MANIFESTS / "train_core.txt"
HPO_MANIFEST = MANIFESTS / "hpo_dev.txt"
ANALYSIS_JSON = FULL / "trial_10_three_seed_stability.json"
ANALYSIS_MD = FULL / "trial_10_three_seed_stability.md"
FINAL_LOCK = FINAL / "FINAL_MODEL_LOCK.json"
ARCHIVED_LOCK = FINAL / "FINAL_MODEL_LOCK_pre_three_seed.json"
FINAL_CHECKPOINT = FINAL / "M2_FINAL.pth"
ARCHIVED_CHECKPOINT = FINAL / "M2_FINAL_pre_three_seed.pth"

TRIAL = 10
THIRD_SEED = 2026
SEEDS = (42, 1337, 2026)
MAX_EPOCHS = 15

# These fixed thresholds are declared before seed-2026 results exist.  They
# are both the warning bounds supplied for the task and the fixed scales used
# for dimensionless median distance (bias magnitude uses a fixed 0.50 m scale).
WARNING_THRESHOLDS = {
    "mae_m": 0.30,
    "rmse_m": 0.40,
    "r2": 0.06,
    "building_mae_m": 0.50,
    "20_50_mae_m": 2.00,
    "full_tile_win_rate_vs_m1": 0.10,
}
MEDIAN_DISTANCE_SCALES = {
    "rmse_m": 0.40,
    "mae_m": 0.30,
    "r2": 0.06,
    "building_mae_m": 0.50,
    "20_50_mae_m": 2.00,
    "absolute_bias_m": 0.50,
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def update_progress(**updates) -> None:
    state = load(PROGRESS_PATH) if PROGRESS_PATH.exists() else {}
    state.update(updates)
    state["updated_at"] = now_iso()
    if state.get("phase_started_unix") is not None:
        state["total_phase_elapsed_seconds"] = time.time() - state["phase_started_unix"]
    dump(PROGRESS_PATH, state)


def _required_existing_paths() -> tuple[Path, ...]:
    return (
        FULL / "trial_10/config.json",
        FULL / "trial_10/training_summary.json",
        FULL / "trial_10/best_model.pth",
        FULL / "trial_10_seed_1337/config.json",
        FULL / "trial_10_seed_1337/training_summary.json",
        FULL / "trial_10_seed_1337/best_model.pth",
        FULL / "final_dev_full_tile/trial_10/metrics.json",
        FULL / "final_dev_full_tile/trial_10/per_tile_metrics.json",
        FULL / "final_dev_full_tile/trial_10_seed_1337/metrics.json",
        FULL / "final_dev_full_tile/trial_10_seed_1337/per_tile_metrics.json",
        FULL / "final_dev_full_tile/m1/per_tile_metrics.json",
        FULL / "final_dev_full_tile/m2_data_control/per_tile_metrics.json",
        ROOT / "baselines/m1_hpo_dev_metrics.json",
        ROOT / "baselines/m1_hpo_dev_per_tile.csv",
        ROOT / "dav2_cache/cache_config.json",
        ROOT / "dav2_cache/cache_manifest.csv",
        TRAIN_MANIFEST,
        HPO_MANIFEST,
        FINAL_MANIFEST,
        FINAL_LOCK,
        FINAL_CHECKPOINT,
    )


def frozen_config(seed: int) -> dict:
    primary = load(FULL / "trial_10/training_summary.json")["config"]
    secondary = load(FULL / "trial_10_seed_1337/training_summary.json")["config"]
    primary_without_seed = {key: value for key, value in primary.items() if key != "seed"}
    secondary_without_seed = {key: value for key, value in secondary.items() if key != "seed"}
    if primary_without_seed != secondary_without_seed:
        raise RuntimeError("seed 42 and seed 1337 training configurations differ beyond seed")
    if primary.get("seed") != 42 or secondary.get("seed") != 1337:
        raise RuntimeError("existing Trial 10 seed identities are not 42 and 1337")
    required = {
        "effective_batch": 16,
        "lr": 0.00027708018817839005,
        "weight_decay": 2.20338284255623e-06,
        "beta": 0.5,
        "lambda_grad": 0.20285517305213446,
        "lambda_log": 0.3471966344674989,
        "weight_10_20": 1.4026222942145867,
        "weight_20_50": 2.076152074393406,
        "weight_50_plus": 5.533736148477046,
        "building_weight": 1.2368551239353343,
        "sampler_random": 36.71746952056274,
        "sampler_building": 34.23564497052743,
        "sampler_gt10": 27.930856547720424,
        "sampler_gt20": 15.360361472968286,
        "sampler_rare_tall": 9.799051363502087,
        "scheduler": "OneCycle",
        "warmup_ratio": 0.0,
        "augmentation_strength": "none",
        "output_parameterization": "softplus",
        "optimizer": "AdamW",
        "epochs": 15,
        "crop_size": 512,
        "loss_type": "height_aware",
        "early_stopping_patience": 0,
        "weight_0_2": 1.0,
        "weight_2_10": 1.0,
    }
    for key, expected in required.items():
        if primary.get(key) != expected:
            raise RuntimeError(f"frozen Trial 10 config mismatch for {key}: {primary.get(key)!r}")
    config = dict(primary)
    config["seed"] = seed
    return config


def validate_inputs(for_launch: bool = False) -> dict:
    missing = [str(path) for path in _required_existing_paths() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required frozen artifacts: {missing}")
    config = frozen_config(THIRD_SEED)
    lock = load(FINAL_LOCK)
    if lock.get("selected_source_trial") != TRIAL or lock.get("selected_seed") != 42:
        raise RuntimeError("current provisional lock is not Trial 10 seed 42")
    hashes = {
        "train": sha256(TRAIN_MANIFEST),
        "hpo_dev": sha256(HPO_MANIFEST),
        "final_dev_val": sha256(FINAL_MANIFEST),
    }
    expected_hashes = {
        "train": lock.get("train_manifest_SHA256"),
        "hpo_dev": lock.get("HPO_dev_manifest_SHA256"),
        "final_dev_val": lock.get("FINAL_DEV_VAL_manifest_SHA256"),
    }
    if hashes != expected_hashes:
        raise RuntimeError(f"frozen manifest hash mismatch: actual={hashes}, expected={expected_hashes}")
    if sha256(FINAL_CHECKPOINT) != lock.get("checkpoint_SHA256"):
        raise RuntimeError("current M2-FINAL checkpoint does not match the current lock")
    if for_launch:
        if WORKER_LOCK.exists():
            raise RuntimeError(f"third-seed worker lock already exists: {WORKER_LOCK}")
        if TRAIN_DIR.exists() or EVAL_DIR.exists() or ANALYSIS_JSON.exists() or ANALYSIS_MD.exists():
            raise RuntimeError("seed-2026 output already exists; refusing to overwrite evidence")
        if ARCHIVED_LOCK.exists() or ARCHIVED_CHECKPOINT.exists():
            raise RuntimeError("pre-three-seed archive already exists; refusing ambiguous rerun")
    return {"config": config, "manifest_sha256": hashes}


def acquire_worker_lock() -> None:
    payload = {
        "pid": os.getpid(),
        "started_at": now_iso(),
        "status": "RUNNING",
        "purpose": "Trial 10 seed 2026 training -> FINAL-DEV-VAL -> three-seed stability",
        "seed": THIRD_SEED,
        "NYC_ALLOWED": False,
    }
    try:
        with WORKER_LOCK.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    except FileExistsError as exc:
        raise RuntimeError(f"third-seed worker lock already exists: {WORKER_LOCK}") from exc


def load_hpo_baseline() -> tuple[dict, dict[str, float]]:
    metrics = load(ROOT / "baselines/m1_hpo_dev_metrics.json")
    with (ROOT / "baselines/m1_hpo_dev_per_tile.csv").open(newline="", encoding="utf-8") as handle:
        tiles = {row["sample_id"]: float(row["mae_m"]) for row in csv.DictReader(handle)}
    return metrics, tiles


def train_third_seed(config: dict, manifest_hashes: dict) -> dict:
    TRAIN_DIR.mkdir(parents=True, exist_ok=False)
    dump(
        TRAIN_DIR / "config.json",
        {
            **config,
            "source_trial": TRIAL,
            "experimental_variable": "seed only",
            "manifest_sha256": manifest_hashes,
            "metrics_during_training": "HPO-DEV PROXY METRICS",
        },
    )
    update_progress(
        workflow="THREE_SEED_STABILITY",
        status="RUNNING",
        phase="SEED_STABILITY_TRAINING",
        phase_started_unix=time.time(),
        worker_pid=os.getpid(),
        trial=TRIAL,
        candidate_trial=TRIAL,
        seed=THIRD_SEED,
        current_epoch=1,
        epoch_total=MAX_EPOCHS,
        output_dir=str(TRAIN_DIR),
        evaluation_status="NOT_STARTED",
        three_seed_analysis_status="NOT_STARTED",
        model_lock_status="PRESERVED_PENDING_STABILITY",
        cuda_available=torch.cuda.is_available(),
        NYC_evaluated=False,
    )
    train_ds = dataset("train_core.txt", True, config["augmentation_strength"])
    val_ds = dataset("hpo_dev.txt", False)
    hpo_metrics, hpo_tiles = load_hpo_baseline()
    set_determinism(THIRD_SEED)
    model = RDAHNetCore(d_model=32, num_heads=4, output_parameterization=config["output_parameterization"])
    summary = train_model(
        model,
        train_ds,
        make_sampler(train_ds, config),
        val_ds,
        config,
        TRAIN_DIR,
        hpo_metrics,
        hpo_tiles,
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    if summary.get("epochs_completed") != MAX_EPOCHS:
        raise RuntimeError(f"seed 2026 completed {summary.get('epochs_completed')} epochs, expected 15")
    del model, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def evaluate_third_seed(config: dict) -> dict:
    def progress_callback(progress: dict) -> None:
        update_progress(
            phase="SEED_STABILITY_FINAL_DEV_VAL",
            evaluation_status="RUNNING",
            evaluation=progress,
            output_dir=str(EVAL_DIR),
        )

    update_progress(
        phase="SEED_STABILITY_FINAL_DEV_VAL",
        evaluation_status="RUNNING",
        evaluation={
            "model_name": "Trial 10 seed 2026",
            "model_index": 1,
            "model_total": 1,
            "tile_number": 0,
            "tile_total": 200,
            "valid_tiles": 0,
            "running_mae_m": None,
        },
        output_dir=str(EVAL_DIR),
    )
    result = evaluate_full_tile_checkpoint(
        model_name="Trial 10 seed 2026",
        checkpoint=TRAIN_DIR / "best_model.pth",
        output_parameterization=config["output_parameterization"],
        manifest=FINAL_MANIFEST,
        out_dir=EVAL_DIR,
        progress_callback=progress_callback,
        model_index=1,
        model_total=1,
    )
    update_progress(evaluation_status="COMPLETE")
    return result


def tile_comparison(model_rows: list[dict], comparator_rows: list[dict]) -> dict:
    model = {row["sample_id"]: row for row in model_rows}
    comparator = {row["sample_id"]: row for row in comparator_rows}
    ids = sorted(set(model) & set(comparator))
    if len(ids) != len(model) or len(ids) != len(comparator) or len(ids) != 200:
        raise RuntimeError("full-tile model/comparator tile sets do not match all 200 tiles")
    delta = np.asarray([model[sid]["mae_m"] - comparator[sid]["mae_m"] for sid in ids], dtype=np.float64)
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


def attach_tile_comparisons(metrics: dict, rows: list[dict], m1_rows: list[dict], control_rows: list[dict]) -> None:
    vs_m1 = tile_comparison(rows, m1_rows)
    vs_control = tile_comparison(rows, control_rows)
    metrics["tile"]["vs_m1"] = vs_m1
    metrics["tile"]["vs_m2_data_control"] = vs_control
    metrics["tile"].update(
        {
            "wins_vs_m1": vs_m1["wins"],
            "losses_vs_m1": vs_m1["losses"],
            "ties_vs_m1": vs_m1["ties"],
            "win_rate_vs_m1": vs_m1["win_rate"],
            "wins_vs_m2_data_control": vs_control["wins"],
            "losses_vs_m2_data_control": vs_control["losses"],
            "ties_vs_m2_data_control": vs_control["ties"],
            "win_rate_vs_m2_data_control": vs_control["win_rate"],
        }
    )


def important_metrics(metrics: dict) -> dict[str, float]:
    return {
        "mae_m": float(metrics["mae_m"]),
        "rmse_m": float(metrics["rmse_m"]),
        "r2": float(metrics["r2"]),
        "pearson_r": float(metrics["pearson_r"]),
        "spearman_rho": float(metrics["spearman_rho"]),
        "bias_m": float(metrics["bias_m"]),
        "building_mae_m": float(metrics["building"]["mae_m"]),
        "10_20_mae_m": float(metrics["height_buckets"]["10-20m"]["mae_m"]),
        "20_50_mae_m": float(metrics["height_buckets"]["20-50m"]["mae_m"]),
        "gte_50_mae_m": float(metrics["height_buckets"][">=50m"]["mae_m"]),
        "full_tile_win_rate_vs_m1": float(metrics["tile"]["vs_m1"]["win_rate"]),
    }


def _summary(values_by_seed: dict[int, float]) -> dict:
    values = np.asarray([values_by_seed[seed] for seed in SEEDS], dtype=np.float64)
    return {
        "values_by_seed": {str(seed): float(values_by_seed[seed]) for seed in SEEDS},
        "mean": float(values.mean()),
        "std_population": float(values.std(ddof=0)),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
    }


def _pair_diagnostics(seed_a: int, seed_b: int, rows: dict[int, dict]) -> dict:
    deltas = {name: abs(rows[seed_a][name] - rows[seed_b][name]) for name in WARNING_THRESHOLDS}
    exceeded = [name for name, delta in deltas.items() if delta > WARNING_THRESHOLDS[name]]
    normalized = sum(deltas[name] / WARNING_THRESHOLDS[name] for name in WARNING_THRESHOLDS) / len(WARNING_THRESHOLDS)
    return {
        "seeds": [seed_a, seed_b],
        "absolute_deltas": deltas,
        "warning_thresholds_exceeded": exceeded,
        "warning_count": len(exceeded),
        "mean_normalized_warning_distance": normalized,
        "tight_pair": not exceeded,
    }


def robust_assessment(rows: dict[int, dict], metric_summaries: dict) -> dict:
    medians = {
        "rmse_m": metric_summaries["rmse_m"]["median"],
        "mae_m": metric_summaries["mae_m"]["median"],
        "r2": metric_summaries["r2"]["median"],
        "building_mae_m": metric_summaries["building_mae_m"]["median"],
        "20_50_mae_m": metric_summaries["20_50_mae_m"]["median"],
        "absolute_bias_m": float(np.median([abs(rows[seed]["bias_m"]) for seed in SEEDS])),
    }
    distances = {}
    components = {}
    for seed in SEEDS:
        values = {
            "rmse_m": rows[seed]["rmse_m"],
            "mae_m": rows[seed]["mae_m"],
            "r2": rows[seed]["r2"],
            "building_mae_m": rows[seed]["building_mae_m"],
            "20_50_mae_m": rows[seed]["20_50_mae_m"],
            "absolute_bias_m": abs(rows[seed]["bias_m"]),
        }
        components[seed] = {
            name: abs(values[name] - medians[name]) / MEDIAN_DISTANCE_SCALES[name]
            for name in MEDIAN_DISTANCE_SCALES
        }
        distances[seed] = float(np.mean(list(components[seed].values())))

    pairwise = {}
    for seed_a, seed_b in ((42, 1337), (42, 2026), (1337, 2026)):
        pairwise[f"{seed_a}_{seed_b}"] = _pair_diagnostics(seed_a, seed_b, rows)

    overall_exceeded = [
        name for name, threshold in WARNING_THRESHOLDS.items() if metric_summaries[name]["range"] > threshold
    ]
    selected_seed = None
    outlier_seed = None
    representative_cluster: list[int] = []
    guard = {}
    classification = "genuinely_seed_sensitive"
    reason = "Multiple important metrics remain separated and no defensible tight two-seed cluster exists."

    if len(overall_exceeded) <= 1:
        classification = "reasonably_stable" if not overall_exceeded else "reasonably_stable_with_warning"
        representative_cluster = list(SEEDS)
        selected_seed = min(SEEDS, key=lambda seed: (distances[seed], rows[seed]["rmse_m"], seed))
        reason = (
            "All three seeds are within every warning threshold; selected the minimum fixed-scale median distance."
            if not overall_exceeded
            else "Only one warning threshold is exceeded; selected the minimum fixed-scale median distance across all seeds."
        )
    else:
        cluster_candidates = []
        for pair in pairwise.values():
            if not pair["tight_pair"]:
                continue
            pair_seeds = pair["seeds"]
            outsider = next(seed for seed in SEEDS if seed not in pair_seeds)
            outsider_pairs = [
                pairwise["_".join(map(str, sorted((outsider, member))))] for member in pair_seeds
            ]
            if all(item["warning_count"] >= 2 for item in outsider_pairs):
                cluster_candidates.append((pair["mean_normalized_warning_distance"], pair_seeds, outsider))
        if cluster_candidates:
            _, representative_cluster, outlier_seed = min(cluster_candidates, key=lambda item: item[0])
            lower_rmse_seed = min(representative_cluster, key=lambda seed: (rows[seed]["rmse_m"], seed))
            peer = next(seed for seed in representative_cluster if seed != lower_rmse_seed)
            guard = {
                "mae_not_materially_worse": rows[lower_rmse_seed]["mae_m"] <= rows[peer]["mae_m"] + WARNING_THRESHOLDS["mae_m"],
                "r2_not_materially_worse": rows[lower_rmse_seed]["r2"] >= rows[peer]["r2"] - WARNING_THRESHOLDS["r2"],
                "building_mae_not_materially_worse": rows[lower_rmse_seed]["building_mae_m"] <= rows[peer]["building_mae_m"] + WARNING_THRESHOLDS["building_mae_m"],
                "20_50_mae_not_materially_worse": rows[lower_rmse_seed]["20_50_mae_m"] <= rows[peer]["20_50_mae_m"] + WARNING_THRESHOLDS["20_50_mae_m"],
            }
            if all(guard.values()):
                selected_seed = lower_rmse_seed
                classification = f"seed_{outlier_seed}_outlier"
                reason = (
                    f"Seeds {representative_cluster[0]} and {representative_cluster[1]} form a threshold-tight cluster; "
                    f"seed {outlier_seed} is separated on at least two warning metrics from both. "
                    f"Within the cluster, seed {selected_seed} has lower RMSE and passes every material-regression guard."
                )
            else:
                reason = "A tight pair was found, but its lower-RMSE checkpoint failed a predefined material-regression guard."
    return {
        "classification": classification,
        "trial_10_seed_sensitive": selected_seed is None,
        "defensible_representative_selection": selected_seed is not None,
        "outlier_seed": outlier_seed,
        "representative_cluster": representative_cluster,
        "selected_representative_seed": selected_seed,
        "reason": reason,
        "overall_warning_thresholds_exceeded": overall_exceeded,
        "median_selection_dimensions": medians,
        "fixed_normalization_scales": MEDIAN_DISTANCE_SCALES,
        "normalized_distance_components": {str(seed): components[seed] for seed in SEEDS},
        "mean_normalized_distance_from_three_seed_median": {str(seed): distances[seed] for seed in SEEDS},
        "pairwise_cluster_diagnostics": pairwise,
        "lower_rmse_cluster_guard": guard,
        "gte_50m_used_as_decider": False,
    }


def self_test_robust_rule() -> None:
    """Exercise stable, outlier, and irreducibly sensitive decision branches."""

    def row(mae, rmse, r2, building, tall, win, bias):
        return {
            "mae_m": mae,
            "rmse_m": rmse,
            "r2": r2,
            "building_mae_m": building,
            "20_50_mae_m": tall,
            "full_tile_win_rate_vs_m1": win,
            "bias_m": bias,
        }

    def assess(rows):
        summaries = {
            name: _summary({seed: rows[seed][name] for seed in SEEDS})
            for name in WARNING_THRESHOLDS
        }
        return robust_assessment(rows, summaries)

    stable = assess(
        {
            42: row(2.80, 5.00, 0.55, 4.00, 10.0, 0.70, -1.0),
            1337: row(2.85, 5.05, 0.54, 4.05, 10.2, 0.72, -1.1),
            2026: row(2.82, 5.02, 0.56, 4.02, 10.1, 0.71, -1.0),
        }
    )
    if stable["selected_representative_seed"] is None or stable["trial_10_seed_sensitive"]:
        raise AssertionError("stable robust-rule branch failed")

    outlier = assess(
        {
            42: row(3.40, 6.00, 0.35, 5.30, 15.0, 0.40, -2.0),
            1337: row(2.80, 5.00, 0.55, 4.00, 10.0, 0.75, -1.0),
            2026: row(2.78, 4.95, 0.56, 3.95, 9.8, 0.76, -0.9),
        }
    )
    if outlier["classification"] != "seed_42_outlier" or outlier["selected_representative_seed"] != 2026:
        raise AssertionError("outlier robust-rule branch failed")

    sensitive = assess(
        {
            42: row(3.50, 6.20, 0.30, 5.50, 16.0, 0.30, -2.5),
            1337: row(2.80, 5.00, 0.55, 4.00, 10.0, 0.70, -1.0),
            2026: row(2.20, 4.20, 0.72, 3.20, 6.0, 0.95, -0.2),
        }
    )
    if sensitive["selected_representative_seed"] is not None or not sensitive["trial_10_seed_sensitive"]:
        raise AssertionError("seed-sensitive robust-rule branch failed")


def make_analysis(metrics_by_seed: dict[int, dict], summaries_by_seed: dict[int, dict]) -> dict:
    m1_rows = load(FULL / "final_dev_full_tile/m1/per_tile_metrics.json")
    control_rows = load(FULL / "final_dev_full_tile/m2_data_control/per_tile_metrics.json")
    for seed in SEEDS:
        key = "trial_10" if seed == 42 else f"trial_10_seed_{seed}"
        rows = load(FULL / "final_dev_full_tile" / key / "per_tile_metrics.json")
        attach_tile_comparisons(metrics_by_seed[seed], rows, m1_rows, control_rows)
        dump(FULL / "final_dev_full_tile" / key / "metrics.json", metrics_by_seed[seed])

    important_by_seed = {seed: important_metrics(metrics_by_seed[seed]) for seed in SEEDS}
    metric_names = list(important_by_seed[42])
    metric_summaries = {
        name: _summary({seed: important_by_seed[seed][name] for seed in SEEDS}) for name in metric_names
    }
    assessment = robust_assessment(important_by_seed, metric_summaries)
    payload = {
        "created_at": now_iso(),
        "trial": TRIAL,
        "seeds": list(SEEDS),
        "best_selected_epochs": {str(seed): int(summaries_by_seed[seed]["best_epoch"]) for seed in SEEDS},
        "important_metrics_by_seed": {str(seed): important_by_seed[seed] for seed in SEEDS},
        "metric_summaries": metric_summaries,
        "warning_thresholds": WARNING_THRESHOLDS,
        "robust_assessment": assessment,
        "automatic_retuning_performed": False,
        "selected_using_NYC": False,
        "tuned_using_NYC": False,
        "calibrated_using_NYC": False,
        "evaluated_on_NYC": False,
        "NYC_statement": "NYC has not been used or evaluated.",
    }
    dump(ANALYSIS_JSON, payload)
    make_markdown(payload)
    return payload


def make_markdown(payload: dict) -> None:
    labels = [
        ("mae_m", "MAE"),
        ("rmse_m", "RMSE"),
        ("r2", "R2"),
        ("pearson_r", "Pearson"),
        ("spearman_rho", "Spearman"),
        ("bias_m", "Bias"),
        ("building_mae_m", "Building MAE"),
        ("10_20_mae_m", "10-20m MAE"),
        ("20_50_mae_m", "20-50m MAE"),
        ("gte_50_mae_m", ">=50m MAE"),
        ("full_tile_win_rate_vs_m1", "Win rate vs M1"),
    ]
    lines = [
        "# Trial 10 Three-Seed Stability",
        "",
        "| Metric | Seed 42 | Seed 1337 | Seed 2026 | Mean | Std | Median | Min | Max | Range |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in labels:
        item = payload["metric_summaries"][key]
        values = item["values_by_seed"]
        lines.append(
            f"| {label} | {values['42']:.6f} | {values['1337']:.6f} | {values['2026']:.6f} | "
            f"{item['mean']:.6f} | {item['std_population']:.6f} | {item['median']:.6f} | "
            f"{item['min']:.6f} | {item['max']:.6f} | {item['range']:.6f} |"
        )
    epochs = payload["best_selected_epochs"]
    assessment = payload["robust_assessment"]
    lines.extend(
        [
            "",
            f"Best selected epochs: seed 42 = {epochs['42']}, seed 1337 = {epochs['1337']}, seed 2026 = {epochs['2026']}.",
            "",
            f"Classification: {assessment['classification']}.",
            "",
            f"Representative seed: {assessment['selected_representative_seed']}.",
            "",
            assessment["reason"],
            "",
            ">=50m was reported but was not used as a deciding metric.",
            "",
            "NYC has not been used or evaluated.",
        ]
    )
    ANALYSIS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_final_lock(analysis: dict, metrics_by_seed: dict[int, dict], summaries_by_seed: dict[int, dict]) -> dict | None:
    assessment = analysis["robust_assessment"]
    selected_seed = assessment["selected_representative_seed"]
    if selected_seed is None:
        return None
    source_dir = FULL / ("trial_10" if selected_seed == 42 else f"trial_10_seed_{selected_seed}")
    source_checkpoint = source_dir / "best_model.pth"
    old_lock = load(FINAL_LOCK)
    old_hash = sha256(FINAL_CHECKPOINT)
    if old_hash != old_lock.get("checkpoint_SHA256"):
        raise RuntimeError("current final checkpoint changed before three-seed lock update")
    if ARCHIVED_LOCK.exists() or ARCHIVED_CHECKPOINT.exists():
        raise RuntimeError("pre-three-seed archive already exists")
    shutil.copy2(FINAL_LOCK, ARCHIVED_LOCK)
    shutil.copy2(FINAL_CHECKPOINT, ARCHIVED_CHECKPOINT)
    if sha256(ARCHIVED_CHECKPOINT) != old_hash:
        raise RuntimeError("archived pre-three-seed checkpoint hash mismatch")

    staged_checkpoint = FINAL / "M2_FINAL.pth.three_seed_next"
    shutil.copy2(source_checkpoint, staged_checkpoint)
    selected_hash = sha256(staged_checkpoint)
    new_lock = dict(old_lock)
    new_lock.update(
        {
            "model_name": "M2-FINAL",
            "selected_source_trial": TRIAL,
            "selected_seed": selected_seed,
            "selected_epoch": int(summaries_by_seed[selected_seed]["best_epoch"]),
            "checkpoint_path": str(FINAL_CHECKPOINT),
            "checkpoint": str(FINAL_CHECKPOINT),
            "checkpoint_SHA256": selected_hash,
            "checkpoint_sha256": selected_hash,
            "checkpoint_creation_timestamp": datetime.fromtimestamp(source_checkpoint.stat().st_mtime).astimezone().isoformat(),
            "final_validation_metrics": metrics_by_seed[selected_seed],
            "three_seed_stability": {
                "seeds": analysis["seeds"],
                "best_selected_epochs": analysis["best_selected_epochs"],
                "seed_metrics": analysis["important_metrics_by_seed"],
                "mean_std_median_ranges": analysis["metric_summaries"],
                "outlier_assessment": assessment,
                "representative_selection_rule": {
                    "median_dimensions": list(MEDIAN_DISTANCE_SCALES),
                    "fixed_normalization_scales": MEDIAN_DISTANCE_SCALES,
                    "cluster_warning_thresholds": WARNING_THRESHOLDS,
                    "choose_lower_rmse_inside_tight_cluster_only_after_material_regression_guards": True,
                    "gte_50m_used_as_decider": False,
                },
                "reason_selected": assessment["reason"],
            },
            "pre_three_seed_lock_path": str(ARCHIVED_LOCK),
            "pre_three_seed_checkpoint_path": str(ARCHIVED_CHECKPOINT),
            "selected_using_NYC": False,
            "tuned_using_NYC": False,
            "calibrated_using_NYC": False,
            "evaluated_on_NYC": False,
            "NYC_statement": "NYC has not been used or evaluated.",
        }
    )
    staged_lock = FINAL / "FINAL_MODEL_LOCK.json.three_seed_next"
    staged_lock.write_text(json.dumps(new_lock, indent=2), encoding="utf-8")
    staged_checkpoint.replace(FINAL_CHECKPOINT)
    staged_lock.replace(FINAL_LOCK)
    (FINAL / "checkpoint.sha256").write_text(f"{selected_hash}  M2_FINAL.pth\n", encoding="utf-8")
    return new_lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    self_test_robust_rule()
    validated = validate_inputs(for_launch=not args.validate_only)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "validated",
                    "trial": TRIAL,
                    "third_seed": THIRD_SEED,
                    "output_dir": str(TRAIN_DIR),
                    "manifest_sha256": validated["manifest_sha256"],
                    "NYC_opened": False,
                },
                indent=2,
            )
        )
        return

    acquire_worker_lock()
    try:
        third_summary = train_third_seed(validated["config"], validated["manifest_sha256"])
        third_result = evaluate_third_seed(validated["config"])
        summaries_by_seed = {
            42: load(FULL / "trial_10/training_summary.json"),
            1337: load(FULL / "trial_10_seed_1337/training_summary.json"),
            2026: third_summary,
        }
        metrics_by_seed = {
            42: load(FULL / "final_dev_full_tile/trial_10/metrics.json"),
            1337: load(FULL / "final_dev_full_tile/trial_10_seed_1337/metrics.json"),
            2026: third_result["metrics"],
        }
        update_progress(phase="THREE_SEED_ANALYSIS", three_seed_analysis_status="RUNNING")
        analysis = make_analysis(metrics_by_seed, summaries_by_seed)
        update_progress(phase="THREE_SEED_MODEL_LOCK", three_seed_analysis_status="COMPLETE")
        lock = update_final_lock(analysis, metrics_by_seed, summaries_by_seed)
        selected_seed = analysis["robust_assessment"]["selected_representative_seed"]
        final_status = "COMPLETE_LOCK_UPDATED" if lock is not None else "STOPPED_SEED_SENSITIVE"
        update_progress(
            status="COMPLETE",
            phase="THREE_SEED_COMPLETE",
            evaluation_status="COMPLETE",
            three_seed_analysis_status="COMPLETE",
            stability_classification=analysis["robust_assessment"]["classification"],
            trial_10_seed_sensitive=analysis["robust_assessment"]["trial_10_seed_sensitive"],
            representative_seed=selected_seed,
            model_lock_status="UPDATED" if lock is not None else "PRESERVED_SEED_SENSITIVE",
            final_checkpoint=lock["checkpoint_path"] if lock is not None else str(FINAL_CHECKPOINT),
            final_checkpoint_sha256=lock["checkpoint_SHA256"] if lock is not None else sha256(FINAL_CHECKPOINT),
            NYC_evaluated=False,
        )
        worker = load(WORKER_LOCK)
        worker.update(
            {
                "status": final_status,
                "completed_at": now_iso(),
                "classification": analysis["robust_assessment"]["classification"],
                "representative_seed": selected_seed,
            }
        )
        dump(WORKER_LOCK, worker)
        print(
            json.dumps(
                {
                    "status": final_status,
                    "third_seed": THIRD_SEED,
                    "best_epoch": third_summary["best_epoch"],
                    "classification": analysis["robust_assessment"]["classification"],
                    "representative_seed": selected_seed,
                    "lock_updated": lock is not None,
                    "NYC_evaluated": False,
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception as exc:
        update_progress(
            status="ERROR",
            error=repr(exc),
            traceback=traceback.format_exc(),
            NYC_evaluated=False,
        )
        worker = load(WORKER_LOCK)
        worker.update({"status": "ERROR", "failed_at": now_iso(), "error": repr(exc)})
        dump(WORKER_LOCK, worker)
        raise


if __name__ == "__main__":
    main()
