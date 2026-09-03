"""Training and deterministic development evaluation engine for Stage A2."""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from src.training.losses import HeightAwareCombinedLoss, RDAHCombinedLoss


BUCKETS = (
    ("0-2m", 0.0, 2.0),
    ("2-10m", 2.0, 10.0),
    ("10-20m", 10.0, 20.0),
    ("20-50m", 20.0, 50.0),
    (">=50m", 50.0, np.inf),
)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if x.size < 10 or np.std(x) <= 1e-8 or np.std(y) <= 1e-8:
        return 0.0
    value = stats.pearsonr(x, y).statistic if method == "pearson" else stats.spearmanr(x, y).statistic
    return float(value) if np.isfinite(value) else 0.0


class RegressionAccumulator:
    """Streaming exact first/second moments plus deterministic quantile samples."""

    def __init__(self, sample_cap: int = 250_000, seed: int = 42):
        self.count = 0
        self.sum_pred = 0.0
        self.sum_gt = 0.0
        self.sum_pred2 = 0.0
        self.sum_gt2 = 0.0
        self.sum_cross = 0.0
        self.sum_error = 0.0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0
        self.sample_cap = sample_cap
        self.rng = np.random.default_rng(seed)
        self.sample_pred = np.empty(sample_cap, dtype=np.float32)
        self.sample_gt = np.empty(sample_cap, dtype=np.float32)
        self.sample_size = 0
        self.seen = 0

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        gt = np.asarray(gt, dtype=np.float64).reshape(-1)
        if not pred.size:
            return
        error = pred - gt
        self.count += pred.size
        self.sum_pred += float(pred.sum())
        self.sum_gt += float(gt.sum())
        self.sum_pred2 += float(np.square(pred).sum())
        self.sum_gt2 += float(np.square(gt).sum())
        self.sum_cross += float((pred * gt).sum())
        self.sum_error += float(error.sum())
        self.sum_abs_error += float(np.abs(error).sum())
        self.sum_sq_error += float(np.square(error).sum())

        # Algorithm R reservoir sampling, vectorised for the initial fill and
        # bounded per update thereafter by sampling at most the remaining cap.
        previous_seen = self.seen
        self.seen += pred.size
        start = 0
        if self.sample_size < self.sample_cap:
            take = min(self.sample_cap - self.sample_size, pred.size)
            end = self.sample_size + take
            self.sample_pred[self.sample_size : end] = pred[:take]
            self.sample_gt[self.sample_size : end] = gt[:take]
            self.sample_size = end
            start = take
            previous_seen += take
        if start < pred.size:
            # Vectorised Algorithm R. Duplicate destination indices within a
            # chunk resolve to their final assignment, an immaterial bounded
            # approximation while retaining uniform inclusion probabilities.
            remaining = pred.size - start
            global_indices = np.arange(previous_seen, previous_seen + remaining, dtype=np.float64)
            replacement = (self.rng.random(remaining) * (global_indices + 1.0)).astype(np.int64)
            keep = replacement < self.sample_cap
            self.sample_pred[replacement[keep]] = pred[start:][keep]
            self.sample_gt[replacement[keep]] = gt[start:][keep]

    def metrics(self, prefix: str = "") -> dict:
        if not self.count:
            return {f"{prefix}pixel_count": 0}
        n = float(self.count)
        mean_pred = self.sum_pred / n
        mean_gt = self.sum_gt / n
        var_pred = max(0.0, self.sum_pred2 / n - mean_pred**2)
        var_gt = max(0.0, self.sum_gt2 / n - mean_gt**2)
        covariance = self.sum_cross / n - mean_pred * mean_gt
        pearson = covariance / math.sqrt(var_pred * var_gt) if var_pred > 0 and var_gt > 0 else 0.0
        r2 = 1.0 - self.sum_sq_error / max(1e-12, self.sum_gt2 - self.sum_gt**2 / n)
        p = self.sample_pred[: self.sample_size].astype(np.float64)
        g = self.sample_gt[: self.sample_size].astype(np.float64)
        residual = p - g
        absolute = np.abs(residual)
        median_residual = float(np.median(residual))
        result = {
            f"{prefix}pixel_count": int(self.count),
            f"{prefix}mae_m": self.sum_abs_error / n,
            f"{prefix}rmse_m": math.sqrt(self.sum_sq_error / n),
            f"{prefix}r2": r2,
            f"{prefix}pearson_r": pearson,
            f"{prefix}spearman_rho": _safe_corr(p, g, "spearman"),
            f"{prefix}bias_m": self.sum_error / n,
            f"{prefix}gt_mean_m": mean_gt,
            f"{prefix}pred_mean_m": mean_pred,
            f"{prefix}median_ae_m": float(np.median(absolute)),
            f"{prefix}nmad_m": float(1.4826 * np.median(np.abs(residual - median_residual))),
            f"{prefix}p50_ae_m": float(np.percentile(absolute, 50)),
            f"{prefix}p75_ae_m": float(np.percentile(absolute, 75)),
            f"{prefix}p90_ae_m": float(np.percentile(absolute, 90)),
            f"{prefix}p95_ae_m": float(np.percentile(absolute, 95)),
            f"{prefix}p99_ae_m": float(np.percentile(absolute, 99)),
            f"{prefix}quantile_sample_size": int(self.sample_size),
        }
        for tolerance in (1, 2, 5, 10):
            result[f"{prefix}within_{tolerance}m_pct"] = float(np.mean(absolute <= tolerance) * 100)
        return result


def _tile_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    diff = pred - gt
    abs_diff = np.abs(diff)
    denominator = float(np.square(gt - gt.mean()).sum())
    return {
        "valid_pixels": int(gt.size),
        "mae_m": float(abs_diff.mean()),
        "rmse_m": float(np.sqrt(np.square(diff).mean())),
        "bias_m": float(diff.mean()),
        "pearson_r": _safe_corr(pred, gt, "pearson"),
        "r2": float(1.0 - np.square(diff).sum() / denominator) if denominator > 1e-12 else None,
    }


def evaluate_model(
    model,
    dataset,
    device: str,
    baseline_tile_mae: dict[str, float] | None = None,
    clamp_nonnegative: bool = True,
    save_predictions: Path | None = None,
) -> tuple[dict, list[dict]]:
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=device == "cuda")
    global_acc = RegressionAccumulator(seed=101)
    building_acc = RegressionAccumulator(seed=102)
    bucket_acc = [RegressionAccumulator(seed=110 + i) for i in range(5)]
    per_tile: list[dict] = []
    prediction_payload: dict[str, np.ndarray] = {}

    with torch.inference_mode():
        for batch in loader:
            sid = batch["sample_id"][0]
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["rel_depth"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                pred_t = model(depth, rgb)
            if clamp_nonnegative:
                pred_t = pred_t.clamp_min(0.0)
            pred = pred_t.squeeze().float().cpu().numpy()
            gt = batch["height"].squeeze().numpy()
            semantic = batch["semantic"].squeeze().numpy()
            mask = batch["valid_mask"].squeeze().numpy().astype(bool)
            valid = mask & np.isfinite(pred) & np.isfinite(gt) & (gt >= 0.0)
            if not np.any(valid):
                continue
            p, g, s = pred[valid], gt[valid], semantic[valid]
            global_acc.update(p, g)
            building_acc.update(p[s == 3], g[s == 3])
            for index, (_, low, high) in enumerate(BUCKETS):
                selected = (g >= low) & (g < high)
                bucket_acc[index].update(p[selected], g[selected])
            tile = {"sample_id": sid, **_tile_metrics(p, g)}
            if baseline_tile_mae is not None and sid in baseline_tile_mae:
                delta = tile["mae_m"] - baseline_tile_mae[sid]
                tile["m1_mae_m"] = baseline_tile_mae[sid]
                tile["mae_delta_vs_m1_m"] = delta
                tile["wins_vs_m1"] = delta < -1e-9
                tile["ties_vs_m1"] = abs(delta) <= 1e-9
            per_tile.append(tile)
            if save_predictions is not None:
                prediction_payload[sid] = pred.astype(np.float16)

    metrics = global_acc.metrics()
    metrics["building"] = building_acc.metrics()
    metrics["height_buckets"] = {
        name: accumulator.metrics() for (name, _, _), accumulator in zip(BUCKETS, bucket_acc)
    }
    tile_mae = np.asarray([row["mae_m"] for row in per_tile], dtype=np.float64)
    metrics["tile"] = {
        "tile_count": len(per_tile),
        "mean_tile_mae_m": float(tile_mae.mean()) if tile_mae.size else None,
        "median_tile_mae_m": float(np.median(tile_mae)) if tile_mae.size else None,
        "std_tile_mae_m": float(tile_mae.std()) if tile_mae.size else None,
        "iqr_tile_mae_m": float(np.percentile(tile_mae, 75) - np.percentile(tile_mae, 25)) if tile_mae.size else None,
        "p90_tile_mae_m": float(np.percentile(tile_mae, 90)) if tile_mae.size else None,
        "p95_tile_mae_m": float(np.percentile(tile_mae, 95)) if tile_mae.size else None,
    }
    if baseline_tile_mae is not None and per_tile:
        wins = sum(bool(row.get("wins_vs_m1")) for row in per_tile)
        ties = sum(bool(row.get("ties_vs_m1")) for row in per_tile)
        metrics["tile"].update(
            {
                "wins_vs_m1": wins,
                "ties_vs_m1": ties,
                "losses_vs_m1": len(per_tile) - wins - ties,
                "win_rate_vs_m1": wins / len(per_tile),
            }
        )
    if save_predictions is not None:
        save_predictions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_predictions, **prediction_payload)
    return metrics, per_tile


def selection_objective(metrics: dict, baseline: dict) -> float:
    """Balanced, dimensionless HPO objective; lower is better."""

    def ratio(value, base, floor=0.05):
        return float(value) / max(float(base), floor)

    bucket, base_bucket = metrics["height_buckets"], baseline["height_buckets"]
    score = (
        0.20 * ratio(metrics["mae_m"], baseline["mae_m"])
        + 0.14 * ratio(metrics["rmse_m"], baseline["rmse_m"])
        + 0.10 * ratio(metrics["building"]["mae_m"], baseline["building"]["mae_m"])
        + 0.09 * ratio(bucket["10-20m"]["mae_m"], base_bucket["10-20m"]["mae_m"])
        + 0.11 * ratio(bucket["20-50m"]["mae_m"], base_bucket["20-50m"]["mae_m"])
        + 0.09 * ratio(bucket[">=50m"]["mae_m"], base_bucket[">=50m"]["mae_m"])
        + 0.07 * ratio(abs(metrics["bias_m"]), abs(baseline["bias_m"]) + 0.25)
        - 0.10 * metrics["r2"]
        - 0.10 * metrics["tile"].get("win_rate_vs_m1", 0.0)
    )
    # Explicit anti-regression constraint for the two dominant low regimes.
    for name in ("0-2m", "2-10m"):
        degradation = ratio(bucket[name]["mae_m"], base_bucket[name]["mae_m"]) - 1.07
        if degradation > 0:
            score += 3.0 * degradation
    if not np.isfinite(score) or abs(metrics["bias_m"]) > 100:
        return float("inf")
    return float(score)


def _make_scheduler(optimizer, scheduler_name: str, epochs: int, steps_per_epoch: int, warmup_ratio: float):
    if scheduler_name == "OneCycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[group["lr"] for group in optimizer.param_groups],
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=max(0.05, warmup_ratio),
        )
        return scheduler, "step"

    if scheduler_name == "CosineWarmup":
        warmup_epochs = max(1, round(epochs * warmup_ratio)) if warmup_ratio else 0

        def schedule(epoch):
            if warmup_epochs and epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule), "epoch"
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6), "epoch"


def train_model(
    model,
    train_dataset,
    train_sampler,
    val_dataset,
    config: dict,
    out_dir: Path,
    baseline_metrics: dict,
    baseline_tile_mae: dict[str, float],
    device: str = "cuda",
    trial=None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    set_determinism(int(config["seed"]))
    model.to(device)
    effective_batch = int(config["effective_batch"])
    physical_batch = max(size for size in range(1, min(8, effective_batch) + 1) if effective_batch % size == 0)
    grad_accum = effective_batch // physical_batch
    loader = DataLoader(
        train_dataset,
        batch_size=physical_batch,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    optimizer_steps_per_epoch = math.ceil(len(loader) / grad_accum)
    scheduler, schedule_unit = _make_scheduler(
        optimizer,
        config["scheduler"],
        int(config["epochs"]),
        optimizer_steps_per_epoch,
        float(config["warmup_ratio"]),
    )
    loss_type = config.get("loss_type", "height_aware")
    if loss_type == "m1":
        criterion = RDAHCombinedLoss(
            beta=float(config["beta"]), grad_weight=float(config["lambda_grad"])
        ).to(device)
    else:
        criterion = HeightAwareCombinedLoss(
            beta=float(config["beta"]),
            grad_weight=float(config["lambda_grad"]),
            log_weight=float(config["lambda_log"]),
            height_weights=(
                float(config.get("weight_0_2", 1.0)),
                float(config.get("weight_2_10", 1.0)),
                float(config["weight_10_20"]),
                float(config["weight_20_50"]),
                float(config["weight_50_plus"]),
            ),
            building_weight=float(config["building_weight"]),
        ).to(device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device == "cuda",
        init_scale=4096.0,
        growth_interval=2000,
    )
    history: list[dict] = []
    sampler_records: list[dict] = []
    best_score = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    amp_skipped_steps = 0
    started = time.time()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, int(config["epochs"]) + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sums = defaultdict(float)
        batch_count = 0
        epoch_sampler = defaultdict(lambda: {"count": 0, "max_gt_sum": 0.0, "building_fraction_sum": 0.0})
        epoch_started = time.time()
        for step, batch in enumerate(loader, 1):
            rgb = batch["rgb"].to(device, non_blocking=True)
            depth = batch["rel_depth"].to(device, non_blocking=True)
            target = batch["height"].to(device, non_blocking=True)
            mask = batch["valid_mask"].to(device, non_blocking=True)
            semantic = batch["semantic"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                prediction = model(depth, rgb)
            if not torch.isfinite(prediction).all():
                finite_fraction = float(torch.isfinite(prediction).float().mean())
                raise FloatingPointError(
                    f"non-finite prediction at epoch={epoch}, step={step}, "
                    f"finite_fraction={finite_fraction:.8f}"
                )
            # Regression reductions stay FP32 even when the network uses AMP.
            with torch.amp.autocast(device_type="cuda", enabled=False):
                if loss_type == "m1":
                    loss, metric_component, gradient_component = criterion(
                        prediction.float(), target.float(), mask
                    )
                    components = {
                        "metric": metric_component,
                        "gradient": gradient_component,
                        "logheight": prediction.sum() * 0.0,
                    }
                else:
                    loss, components = criterion(
                        prediction.float(), target.float(), mask, semantic
                    )
                scaled_loss = loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch={epoch}, step={step}")
            scaler.scale(scaled_loss).backward()
            if step % grad_accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not torch.isfinite(grad_norm):
                    if not scaler.is_enabled():
                        raise FloatingPointError(
                            f"non-finite gradient at epoch={epoch}, step={step}"
                        )
                    # unscale_ has already recorded the non-finite gradient;
                    # scaler.step therefore skips this optimizer update and
                    # scaler.update lowers the scale.  This is normal AMP
                    # recovery, not a failed HPO trial.
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_ran = False
                    amp_skipped_steps += 1
                else:
                    scale_before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_ran = scaler.get_scale() >= scale_before
                if schedule_unit == "step" and optimizer_ran:
                    scheduler.step()
            loss_sums["total"] += float(loss.detach())
            for name, value in components.items():
                loss_sums[name] += float(value.detach())
            batch_count += 1
            categories = batch["sampling_category"]
            max_values = batch["crop_max_gt"].tolist()
            building_values = batch["building_fraction"].tolist()
            for category, max_gt, building_fraction in zip(categories, max_values, building_values):
                record = epoch_sampler[category]
                record["count"] += 1
                record["max_gt_sum"] += float(max_gt)
                record["building_fraction_sum"] += float(building_fraction)
        if schedule_unit == "epoch":
            scheduler.step()

        metrics, _ = evaluate_model(
            model,
            val_dataset,
            device,
            baseline_tile_mae=baseline_tile_mae,
            clamp_nonnegative=True,
        )
        score = selection_objective(metrics, baseline_metrics)
        realized_sampler = {}
        for category, values in epoch_sampler.items():
            realized_sampler[category] = {
                "count": values["count"],
                "ratio": values["count"] / max(1, len(train_sampler)),
                "mean_max_gt_m": values["max_gt_sum"] / values["count"],
                "mean_building_fraction": values["building_fraction_sum"] / values["count"],
            }
        record = {
            "epoch": epoch,
            "train_loss": loss_sums["total"] / max(1, batch_count),
            "metric_loss": loss_sums["metric"] / max(1, batch_count),
            "gradient_loss": loss_sums["gradient"] / max(1, batch_count),
            "logheight_loss": loss_sums["logheight"] / max(1, batch_count),
            "objective": score,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.time() - epoch_started,
            "candidate_elapsed_seconds": time.time() - started,
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
            ),
            "amp_skipped_steps_total": amp_skipped_steps,
            "non_finite_count": 0,
            "realized_sampler_distribution": realized_sampler,
            "metrics": metrics,
        }
        history.append(record)
        for category, values in epoch_sampler.items():
            sampler_records.append(
                {
                    "epoch": epoch,
                    "category": category,
                    "count": values["count"],
                    "mean_max_gt_m": values["max_gt_sum"] / values["count"],
                    "mean_building_fraction": values["building_fraction_sum"] / values["count"],
                }
            )
        print(
            f"epoch={epoch}/{config['epochs']} loss={record['train_loss']:.4f} "
            f"dev_mae={metrics['mae_m']:.4f} r2={metrics['r2']:+.4f} "
            f"win={metrics['tile'].get('win_rate_vs_m1', 0):.3f} objective={score:.5f} "
            f"seconds={record['epoch_seconds']:.1f}",
            flush=True,
        )
        if score < best_score - 1e-6:
            best_score, best_epoch, epochs_without_improvement = score, epoch, 0
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            (out_dir / "best_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        else:
            epochs_without_improvement += 1
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "best_objective": best_score,
            },
            out_dir / "latest_checkpoint.pth",
        )
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned(f"pruned at epoch {epoch}: objective={score}")
        patience = int(config.get("early_stopping_patience", 0))
        if patience and epochs_without_improvement >= patience:
            break

    import csv

    if sampler_records:
        with (out_dir / "realized_sampler_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(sampler_records[0]))
            writer.writeheader()
            writer.writerows(sampler_records)
        (out_dir / "sampler_stats.json").write_text(
            json.dumps(sampler_records, indent=2), encoding="utf-8"
        )
    summary = {
        "best_objective": best_score,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "runtime_seconds": time.time() - started,
        "amp_skipped_steps_total": amp_skipped_steps,
        "peak_vram_mb": (
            torch.cuda.max_memory_allocated() / (1024**2) if device == "cuda" else 0.0
        ),
        "config": config,
        "best_metrics": json.loads((out_dir / "best_metrics.json").read_text(encoding="utf-8")),
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "runtime_stats.json").write_text(
        json.dumps(
            {
                "runtime_seconds": summary["runtime_seconds"],
                "epochs_completed": summary["epochs_completed"],
                "seconds_per_epoch": summary["runtime_seconds"] / max(1, summary["epochs_completed"]),
                "peak_vram_mb": summary["peak_vram_mb"],
                "amp_skipped_steps_total": amp_skipped_steps,
                "device": device,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary
