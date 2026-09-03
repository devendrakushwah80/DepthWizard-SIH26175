"""Run Stage A2 control, persistent Optuna HPO, and PHL/DC-only selection."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
from optuna.trial import TrialState

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.rdah_net import RDAHNetCore
from src.training.stage_a2 import GAMUSStageA2Dataset, HeightAwareCropSampler
from src.training.stage_a2_engine import evaluate_model, selection_objective, train_model


ROOT = Path("outputs/player1_stage_a2")
MANIFESTS = Path("data/gamus/splits/stage_a2")
DATA_DIR = Path("data/gamus/dataset")
CACHE_DIR = Path("data/gamus/cache_dav2")
CROP_INDEX = ROOT / "data_analysis/crop_index.csv"
M1_CHECKPOINT = Path("outputs/player1_stage_a/a1_rdah/checkpoints/best_mae_model.pth")


def dump_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def dataset(manifest: str, training: bool, augmentation: str = "none"):
    return GAMUSStageA2Dataset(
        MANIFESTS / manifest,
        DATA_DIR,
        CACHE_DIR,
        crop_size=(512, 512),
        is_training=training,
        augmentation_strength=augmentation,
        require_depth=True,
    )


def load_model(checkpoint: Path, output_parameterization: str = "unconstrained"):
    model = RDAHNetCore(
        d_model=32,
        num_heads=4,
        output_parameterization=output_parameterization,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    return model


def ensure_baselines(force: bool = False) -> tuple[dict, dict[str, float], dict, dict[str, float]]:
    baseline_dir = ROOT / "baselines"
    paths = {
        "hpo_metrics": baseline_dir / "m1_hpo_dev_metrics.json",
        "hpo_tiles": baseline_dir / "m1_hpo_dev_per_tile.csv",
        "final_metrics": baseline_dir / "m1_final_dev_metrics.json",
        "final_tiles": baseline_dir / "m1_final_dev_per_tile.csv",
    }
    if force or not all(path.exists() for path in paths.values()):
        model = load_model(M1_CHECKPOINT).to(device_name())
        for split, manifest in (("hpo", "hpo_dev.txt"), ("final", "final_dev_val.txt")):
            metrics, tiles = evaluate_model(model, dataset(manifest, False), device_name())
            dump_json(paths[f"{split}_metrics"], metrics)
            write_csv(paths[f"{split}_tiles"], tiles)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    hpo_metrics = json.loads(paths["hpo_metrics"].read_text(encoding="utf-8"))
    final_metrics = json.loads(paths["final_metrics"].read_text(encoding="utf-8"))

    def tile_map(path: Path) -> dict[str, float]:
        with path.open(newline="", encoding="utf-8") as handle:
            return {row["sample_id"]: float(row["mae_m"]) for row in csv.DictReader(handle)}

    return hpo_metrics, tile_map(paths["hpo_tiles"]), final_metrics, tile_map(paths["final_tiles"])


def ratios_from_config(config: dict) -> dict[str, float]:
    raw = {
        "random": float(config["sampler_random"]),
        "building_rich": float(config["sampler_building"]),
        "gt10": float(config["sampler_gt10"]),
        "gt20": float(config["sampler_gt20"]),
        "rare_tall": float(config["sampler_rare_tall"]),
    }
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def make_sampler(train_ds, config):
    return HeightAwareCropSampler(
        train_ds,
        CROP_INDEX,
        ratios=ratios_from_config(config),
        seed=int(config["seed"]),
        samples_per_epoch=len(train_ds),
    )


def data_control() -> dict:
    out_dir = ROOT / "m2_data_control"
    summary_path = out_dir / "training_summary.json"
    hpo_metrics, hpo_tiles, _, _ = ensure_baselines()
    if summary_path.exists() and (out_dir / "best_model.pth").exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    config = {
        "seed": 42,
        "epochs": 15,
        "effective_batch": 8,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "beta": 1.0,
        "lambda_grad": 0.5,
        "lambda_log": 0.0,
        "weight_0_2": 1.0,
        "weight_2_10": 1.0,
        "weight_10_20": 1.0,
        "weight_20_50": 1.0,
        "weight_50_plus": 1.0,
        "building_weight": 1.0,
        "sampler_random": 100.0,
        "sampler_building": 0.0,
        "sampler_gt10": 0.0,
        "sampler_gt20": 0.0,
        "sampler_rare_tall": 0.0,
        "scheduler": "Cosine",
        "warmup_ratio": 0.0,
        "augmentation_strength": "mild",
        "output_parameterization": "unconstrained",
        "early_stopping_patience": 4,
        "loss_type": "m1",
        "control_definition": "M1 architecture/loss/random sampling; only training population enlarged",
    }
    train_ds = dataset("train_core.txt", True, config["augmentation_strength"])
    val_ds = dataset("hpo_dev.txt", False)
    model = RDAHNetCore(d_model=32, num_heads=4, output_parameterization="unconstrained")
    summary = train_model(
        model,
        train_ds,
        make_sampler(train_ds, config),
        val_ds,
        config,
        out_dir,
        hpo_metrics,
        hpo_tiles,
        device_name(),
    )
    del model, train_ds, val_ds
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary


def suggest_config(trial: optuna.Trial, epochs: int = 5, seed: int = 42) -> dict:
    return {
        "seed": seed,
        "epochs": epochs,
        "effective_batch": trial.suggest_categorical("effective_batch", [4, 8, 12, 16]),
        "lr": trial.suggest_float("lr", 3e-5, 3e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 5e-4, log=True),
        "beta": trial.suggest_categorical("beta", [0.5, 1.0, 2.0, 4.0]),
        "lambda_grad": trial.suggest_float("lambda_grad", 0.2, 1.0),
        "lambda_log": trial.suggest_float("lambda_log", 0.0, 0.5),
        "weight_0_2": 1.0,
        "weight_2_10": 1.0,
        "weight_10_20": trial.suggest_float("weight_10_20", 1.25, 3.0),
        "weight_20_50": trial.suggest_float("weight_20_50", 2.0, 5.0),
        "weight_50_plus": trial.suggest_float("weight_50_plus", 3.0, 8.0),
        "building_weight": trial.suggest_float("building_weight", 1.0, 2.5),
        "sampler_random": trial.suggest_float("sampler_random", 20.0, 40.0),
        "sampler_building": trial.suggest_float("sampler_building", 20.0, 35.0),
        "sampler_gt10": trial.suggest_float("sampler_gt10", 15.0, 30.0),
        "sampler_gt20": trial.suggest_float("sampler_gt20", 10.0, 25.0),
        "sampler_rare_tall": trial.suggest_float("sampler_rare_tall", 5.0, 15.0),
        "scheduler": trial.suggest_categorical("scheduler", ["Cosine", "CosineWarmup", "OneCycle"]),
        "warmup_ratio": trial.suggest_categorical("warmup_ratio", [0.0, 0.05, 0.10]),
        "augmentation_strength": trial.suggest_categorical("augmentation_strength", ["none", "mild", "moderate"]),
        "output_parameterization": trial.suggest_categorical(
            "output_parameterization", ["unconstrained", "softplus"]
        ),
        "optimizer": "AdamW",
        "crop_size": 512,
        "early_stopping_patience": 0,
        "loss_type": "height_aware",
    }


def config_from_params(params: dict, epochs: int, seed: int) -> dict:
    config = dict(params)
    config.update(
        {
            "seed": seed,
            "epochs": epochs,
            "weight_0_2": 1.0,
            "weight_2_10": 1.0,
            "optimizer": "AdamW",
            "crop_size": 512,
            "early_stopping_patience": 4,
            "loss_type": "height_aware",
        }
    )
    return config


def create_study() -> optuna.Study:
    hpo_dir = ROOT / "hpo"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(hpo_dir / 'optuna_stage_a2.db').resolve().as_posix()}"
    return optuna.create_study(
        study_name="depthwizard_stage_a2",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2, interval_steps=1),
    )


def objective_factory(hpo_metrics, hpo_tiles):
    def objective(trial: optuna.Trial) -> float:
        trial_dir = ROOT / "hpo/trial_logs" / f"trial_{trial.number:03d}"
        config = suggest_config(trial)
        dump_json(trial_dir / "config.json", {**config, "normalized_sampler_ratios": ratios_from_config(config)})
        model = None
        try:
            train_ds = dataset("train_core.txt", True, config["augmentation_strength"])
            val_ds = dataset("hpo_dev.txt", False)
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
                trial_dir,
                hpo_metrics,
                hpo_tiles,
                device_name(),
                trial=trial,
            )
            trial.set_user_attr("best_epoch", summary["best_epoch"])
            trial.set_user_attr("best_metrics", summary["best_metrics"])
            return float(summary["best_objective"])
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return objective


def save_hpo_outputs(study: optuna.Study) -> None:
    hpo_dir = ROOT / "hpo"
    study.trials_dataframe().to_csv(hpo_dir / "trials.csv", index=False)
    completed = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed:
        return
    best = study.best_trial
    best_config = config_from_params(best.params, epochs=15, seed=42)
    dump_json(
        hpo_dir / "best_params.json",
        {
            "trial_number": best.number,
            "objective": best.value,
            "params": best.params,
            "normalized_sampler_ratios": ratios_from_config(best_config),
            "best_metrics": best.user_attrs.get("best_metrics"),
        },
    )
    plots_dir = hpo_dir / "plots"
    slice_dir = hpo_dir / "slice_plots"
    plots_dir.mkdir(exist_ok=True)
    slice_dir.mkdir(exist_ok=True)
    plot_jobs = [
        ("optimization_history.png", lambda: optuna.visualization.matplotlib.plot_optimization_history(study)),
        ("parameter_importance.png", lambda: optuna.visualization.matplotlib.plot_param_importances(study)),
        ("parallel_coordinate.png", lambda: optuna.visualization.matplotlib.plot_parallel_coordinate(study)),
    ]
    for filename, function in plot_jobs:
        try:
            axis = function()
            axis.figure.tight_layout()
            axis.figure.savefig(plots_dir / filename, dpi=160, bbox_inches="tight")
            axis.figure.savefig(hpo_dir / filename, dpi=160, bbox_inches="tight")
            plt.close(axis.figure)
        except Exception as exc:
            dump_json(plots_dir / f"{filename}.error.json", {"error": repr(exc)})
    for parameter in best.params:
        try:
            axes = optuna.visualization.matplotlib.plot_slice(study, params=[parameter])
            figure = axes.figure if hasattr(axes, "figure") else np.asarray(axes).flat[0].figure
            figure.tight_layout()
            figure.savefig(slice_dir / f"{parameter}.png", dpi=140, bbox_inches="tight")
            plt.close(figure)
        except Exception as exc:
            dump_json(slice_dir / f"{parameter}.error.json", {"error": repr(exc)})
    try:
        importance = optuna.importance.get_param_importances(study)
    except Exception as exc:
        importance = {"error": repr(exc)}
    dump_json(hpo_dir / "parameter_importance.json", importance)


def run_hpo(total_trials: int) -> optuna.Study:
    hpo_metrics, hpo_tiles, _, _ = ensure_baselines()
    study = create_study()
    remaining = max(0, total_trials - len(study.trials))
    if remaining:
        study.optimize(
            objective_factory(hpo_metrics, hpo_tiles),
            n_trials=remaining,
            gc_after_trial=True,
            catch=(RuntimeError, FloatingPointError, FileNotFoundError),
        )
    save_hpo_outputs(study)
    return study


def select_top_trials(study: optuna.Study, count: int = 3) -> list[optuna.trial.FrozenTrial]:
    complete = sorted(
        [trial for trial in study.trials if trial.state == TrialState.COMPLETE and trial.value is not None],
        key=lambda trial: trial.value,
    )
    return complete[:count]


def evaluate_checkpoint(
    checkpoint: Path,
    config: dict,
    metrics_path: Path,
    tiles_path: Path,
    final_baseline_tiles: dict[str, float],
) -> dict:
    if metrics_path.exists() and tiles_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    model = load_model(checkpoint, config["output_parameterization"]).to(device_name())
    metrics, tiles = evaluate_model(
        model,
        dataset("final_dev_val.txt", False),
        device_name(),
        baseline_tile_mae=final_baseline_tiles,
    )
    dump_json(metrics_path, metrics)
    write_csv(tiles_path, tiles)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def full_training(top_count: int = 3, multi_seed_count: int = 2) -> list[dict]:
    study = create_study()
    top_trials = select_top_trials(study, top_count)
    if len(top_trials) < top_count:
        raise RuntimeError(f"only {len(top_trials)} completed HPO trials; need {top_count}")
    hpo_metrics, hpo_tiles, final_baseline, final_tiles = ensure_baselines()
    results: list[dict] = []

    # Include the data-scale control in the independent selection table.
    control = data_control()
    control_metrics = evaluate_checkpoint(
        ROOT / "m2_data_control/best_model.pth",
        control["config"],
        ROOT / "m2_data_control/final_dev_metrics.json",
        ROOT / "m2_data_control/final_dev_per_tile.csv",
        final_tiles,
    )
    results.append(
        {
            "candidate": "M2-DATA-CONTROL",
            "trial_number": None,
            "seed": control["config"]["seed"],
            "checkpoint": str(ROOT / "m2_data_control/best_model.pth"),
            "config": control["config"],
            "hpo_dev_metrics": control["best_metrics"],
            "final_dev_metrics": control_metrics,
            "final_objective": selection_objective(control_metrics, final_baseline),
        }
    )

    for rank, trial in enumerate(top_trials, 1):
        seeds = [42, 1337] if rank <= multi_seed_count else [42]
        for seed in seeds:
            run_dir = ROOT / "full_training" / f"candidate_{rank}" / f"seed_{seed}"
            config = config_from_params(trial.params, epochs=15, seed=seed)
            summary_path = run_dir / "training_summary.json"
            if summary_path.exists() and (run_dir / "best_model.pth").exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            else:
                train_ds = dataset("train_core.txt", True, config["augmentation_strength"])
                val_ds = dataset("hpo_dev.txt", False)
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
                    run_dir,
                    hpo_metrics,
                    hpo_tiles,
                    device_name(),
                )
                del model, train_ds, val_ds
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            final_metrics = evaluate_checkpoint(
                run_dir / "best_model.pth",
                config,
                run_dir / "final_dev_metrics.json",
                run_dir / "final_dev_per_tile.csv",
                final_tiles,
            )
            results.append(
                {
                    "candidate": f"candidate_{rank}",
                    "trial_number": trial.number,
                    "seed": seed,
                    "checkpoint": str(run_dir / "best_model.pth"),
                    "config": config,
                    "hpo_dev_metrics": summary["best_metrics"],
                    "final_dev_metrics": final_metrics,
                    "final_objective": selection_objective(final_metrics, final_baseline),
                }
            )
    dump_json(ROOT / "full_training/candidate_results.json", results)

    flat = []
    for row in results:
        metrics = row["final_dev_metrics"]
        flat.append(
            {
                "candidate": row["candidate"],
                "trial_number": row["trial_number"],
                "seed": row["seed"],
                "objective": row["final_objective"],
                "mae_m": metrics["mae_m"],
                "rmse_m": metrics["rmse_m"],
                "r2": metrics["r2"],
                "pearson_r": metrics["pearson_r"],
                "bias_m": metrics["bias_m"],
                "building_mae_m": metrics["building"]["mae_m"],
                "10_20_mae_m": metrics["height_buckets"]["10-20m"]["mae_m"],
                "20_50_mae_m": metrics["height_buckets"]["20-50m"]["mae_m"],
                "50_plus_mae_m": metrics["height_buckets"][">=50m"]["mae_m"],
                "tile_win_rate": metrics["tile"].get("win_rate_vs_m1"),
                "checkpoint": row["checkpoint"],
            }
        )
    write_csv(ROOT / "full_training/candidate_results.csv", flat)
    return results


def lock_final_model(results: list[dict] | None = None) -> dict:
    if results is None:
        results = json.loads((ROOT / "full_training/candidate_results.json").read_text(encoding="utf-8"))
    _, _, final_baseline, _ = ensure_baselines()
    non_control = [row for row in results if row["candidate"] != "M2-DATA-CONTROL"]
    grouped: dict[str, list[dict]] = {}
    for row in non_control:
        grouped.setdefault(row["candidate"], []).append(row)
    stability = []
    for candidate, runs in grouped.items():
        objectives = np.asarray([row["final_objective"] for row in runs], dtype=float)
        stability.append(
            {
                "candidate": candidate,
                "mean_objective": float(objectives.mean()),
                "std_objective": float(objectives.std()),
                "runs": len(runs),
            }
        )
    stability.sort(key=lambda row: row["mean_objective"])
    winning_config = stability[0]["candidate"]
    candidate_runs = grouped[winning_config]
    mean_score = stability[0]["mean_objective"]
    # Choose the representative run closest to its multi-seed mean, preventing
    # promotion of an obviously lucky seed.
    selected = min(candidate_runs, key=lambda row: (abs(row["final_objective"] - mean_score), row["seed"]))

    final_dir = ROOT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    destination = final_dir / "M2_FINAL.pth"
    shutil.copy2(selected["checkpoint"], destination)
    checkpoint_hash = sha256(destination)
    (final_dir / "checkpoint.sha256").write_text(f"{checkpoint_hash}  M2_FINAL.pth\n", encoding="utf-8")
    audit = json.loads((ROOT / "data_audit.json").read_text(encoding="utf-8"))
    lock = {
        "model_name": "M2-FINAL",
        "checkpoint": str(destination),
        "checkpoint_sha256": checkpoint_hash,
        "selected_using": "PHL_DC_ONLY",
        "nyc_used_for_hpo": False,
        "nyc_used_for_model_selection": False,
        "train_manifest_sha256": audit["manifest_sha256"]["train_core"],
        "dev_manifest_sha256": audit["manifest_sha256"]["final_dev_val"],
        "hpo_dev_manifest_sha256": audit["manifest_sha256"]["hpo_dev"],
        "hyperparameters": selected["config"],
        "validation_metrics": selected["final_dev_metrics"],
        "selection_objective": selected["final_objective"],
        "selected_candidate": selected["candidate"],
        "selected_seed": selected["seed"],
        "multi_seed_stability": stability,
        "m1_final_dev_metrics": final_baseline,
        "calibration": None,
        "tta": None,
        "ensemble": None,
        "partial_dav2_finetuning": "not_run_compute_budget_and_cached-prior_primary_protocol",
    }
    dump_json(final_dir / "FINAL_MODEL_LOCK.json", lock)
    return lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("baseline", "data_control", "hpo", "full", "lock", "all"),
        default="all",
    )
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    started = time.time()
    if args.phase == "baseline":
        ensure_baselines(force=True)
    elif args.phase == "all":
        ensure_baselines()
    if args.phase in {"data_control", "all"}:
        data_control()
    if args.phase in {"hpo", "all"}:
        run_hpo(args.trials)
    results = None
    if args.phase in {"full", "all"}:
        results = full_training()
    if args.phase in {"lock", "all"}:
        lock_final_model(results)
    print(f"stage_a2_development phase={args.phase} elapsed_seconds={time.time()-started:.1f}")


if __name__ == "__main__":
    main()
