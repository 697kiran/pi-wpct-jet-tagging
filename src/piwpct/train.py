"""Shared training loop and regime execution engine.

Supports:
- All 6 regimes (Baseline CE, CE+TempScale, CE+Mondrian, ConfTr, WPCT, PI-WPCT)
- All 4 backbones (ParticleNet-Lite, LorentzNet-small, ParT, GBT)
- Periodic non-differentiable validation risk refresh (PI-WPCT only)
- Complete checkpoint metadata (config, git commit, seed, manifest hash)
- Leakage isolation guards
"""

import copy
import os
import time
from typing import Any, Dict, Optional, Tuple, cast
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from piwpct.backbones import build_backbone
from piwpct.backbones.gbt import GBTModel
from piwpct.losses import BaselineCELoss, ConfTrLoss, PIWPCTLoss, WPCTLoss
from piwpct.risk_allocator import RiskAllocator
from piwpct.utils.logging_utils import EpochMetricsLogger, TrajectoryLogger, get_git_commit_hash, setup_logger
from piwpct.utils.run_directory import (
    create_run_directory,
    manifest_identifier,
    update_run_metadata,
)
from piwpct.utils.split_integrity import SplitAuditor, SplitIntegrityViolation


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling fitted on validation logits."""

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / torch.clamp(self.temperature, min=1e-6)

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        logits_t = torch.tensor(val_logits, dtype=torch.float32)
        labels_t = torch.tensor(val_labels, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        def _eval():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits_t), labels_t)
            loss.backward()
            return loss

        optimizer.step(_eval)
        with torch.no_grad():
            self.temperature.clamp_(min=1e-6)
        return float(self.temperature.item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply one validated scalar temperature to logits for any evaluation split."""
    value = float(temperature)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"Temperature must be positive and finite, received {temperature!r}.")
    return np.asarray(logits) / value


def scale_calibration_and_test_logits(
    calibration_logits: np.ndarray,
    test_logits: np.ndarray,
    temperature: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the same fitted transform before conformal calibration and test scoring."""
    return apply_temperature(calibration_logits, temperature), apply_temperature(test_logits, temperature)


def build_loss(regime: str, config: Dict[str, Any], target_wp: float) -> nn.Module:
    """Instantiate loss function for the specified regime."""
    reg = regime.lower()
    reg_cfg = config.get("regime", {})
    loss_cfg = reg_cfg.get("loss", {})

    if reg in {"baseline_ce", "ce_temp_scaling", "ce_mondrian_cp"}:
        return BaselineCELoss()
    elif reg == "conftr":
        return ConfTrLoss(
            alpha=loss_cfg.get("alpha", 0.10),
            temperature=loss_cfg.get("temperature", 0.1),
            coverage_penalty_weight=loss_cfg.get("coverage_penalty_weight", 1.0),
            size_penalty_weight=loss_cfg.get("size_penalty_weight", 0.1),
        )
    elif reg == "wpct":
        return WPCTLoss(
            target_wp=target_wp,
            lambda_eff=loss_cfg.get("lambda_eff", 2.0),
            lambda_bg=loss_cfg.get("lambda_bg", 1.0),
            lambda_light=loss_cfg.get("lambda_light", 0.7),
            lambda_charm=loss_cfg.get("lambda_charm", 0.3),
            temperature=loss_cfg.get("temperature", 0.05),
        )
    elif reg == "pi_wpct":
        return PIWPCTLoss(
            target_wp=target_wp,
            lambda_eff=loss_cfg.get("lambda_eff", 2.0),
            lambda_bg=loss_cfg.get("lambda_bg", 1.0),
            initial_lambda_light=loss_cfg.get("initial_lambda_light", 0.5),
            initial_lambda_charm=loss_cfg.get("initial_lambda_charm", 0.5),
            temperature=loss_cfg.get("temperature", 0.05),
        )
    else:
        raise ValueError(f"Unknown regime: {regime}")


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    target_wp: float,
) -> Dict[str, float]:
    """Execute one training epoch using gradient optimization on train split only."""
    # Strict leakage assertion
    if hasattr(train_loader.dataset, "split_name"):
        split_name = train_loader.dataset.split_name.lower()
        if split_name in {"calibration", "cal"}:
            raise SplitIntegrityViolation("FATAL: Calibration split used in train_epoch!")
        if split_name in {"test", "testing"}:
            raise SplitIntegrityViolation("FATAL: Test split used in train_epoch!")

    model.train()
    total_loss = 0.0
    metric_sums: Dict[str, float] = {}
    n_batches = 0

    for batch in train_loader:
        features = batch["features"].to(device)
        points = batch["points"].to(device) if "points" in batch else None
        mask = batch["mask"].to(device) if "mask" in batch else None
        vectors = batch["vectors"].to(device) if "vectors" in batch else None
        high_level = batch["high_level"].to(device) if "high_level" in batch else None
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(
            features=features,
            points=points,
            mask=mask,
            vectors=vectors,
            high_level=high_level,
        )
        loss, step_metrics = loss_fn(logits, labels, target_wp=target_wp)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        for k, v in step_metrics.items():
            metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
        n_batches += 1

    avg_metrics = {k: v / max(n_batches, 1) for k, v in metric_sums.items()}
    avg_metrics["train_loss"] = total_loss / max(n_batches, 1)
    return avg_metrics


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    target_wp: float,
    collect_outputs: bool = True,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on validation split."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits, all_labels = [], []

    for batch in val_loader:
        features = batch["features"].to(device)
        points = batch["points"].to(device) if "points" in batch else None
        mask = batch["mask"].to(device) if "mask" in batch else None
        vectors = batch["vectors"].to(device) if "vectors" in batch else None
        high_level = batch["high_level"].to(device) if "high_level" in batch else None
        labels = batch["labels"].to(device)

        logits = model(
            features=features,
            points=points,
            mask=mask,
            vectors=vectors,
            high_level=high_level,
        )
        loss, _ = loss_fn(logits, labels, target_wp=target_wp)
        total_loss += float(loss.item())
        n_batches += 1

        if collect_outputs:
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    logits_arr = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, 0), dtype=np.float32)
    labels_arr = np.concatenate(all_labels, axis=0) if all_labels else np.empty((0,), dtype=np.int64)
    return {"val_loss": total_loss / max(n_batches, 1)}, logits_arr, labels_arr


def _train_pipeline_in_run(
    regime: str,
    backbone_name: str,
    config: Dict[str, Any],
    dataloaders: Dict[str, DataLoader],
    target_wp: float = 0.70,
    seed: int = 42,
    device_str: Optional[str] = None,
    *,
    output_dir: str,
    run_id: str,
) -> Dict[str, Any]:
    """Execute end-to-end training pipeline for a given regime and backbone."""
    run_dir = output_dir
    logger = setup_logger(
        f"train.{regime}.{backbone_name}",
        log_file=os.path.join(run_dir, "logs", "application.log"),
    )
    logger.info(f"Starting run: run_id={run_id}, regime={regime}, backbone={backbone_name}, wp={target_wp}, seed={seed}")

    train_dataset = dataloaders["train"].dataset
    target_mode = getattr(train_dataset, "target_mode", "flavour_tagging")
    class_names = list(getattr(train_dataset, "class_names", ["light", "charm", "b"]))
    if target_mode == "process_classification" and regime in {"wpct", "pi_wpct"}:
        raise ValueError(
            "WPCT/PI-WPCT use b-efficiency and light/charm mistag semantics; they cannot be applied "
            "to JetClass process classes. Use a generic classification regime or a valid flavour target."
        )

    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    trajectory_path = os.path.join(run_dir, "trajectories", "trajectories.jsonl")
    trajectory_logger = TrajectoryLogger(trajectory_path, run_id=run_id)

    # Instantiate backbone
    backbone_cfg = dict(config.get("backbone", {}))
    if target_mode == "process_classification":
        backbone_cfg["num_classes"] = len(class_names)
    model = build_backbone(backbone_name, backbone_cfg).to(device)

    # If backbone is GBT, handle tree training directly
    if isinstance(model, GBTModel):
        if not hasattr(train_dataset, "data"):
            raise ValueError(
                "GBT training currently needs dense high-level arrays and is not available for "
                "memory-mapped jagged real-data shards. Choose a neural backbone."
            )
        logger.info("Fitting Gradient Boosted Trees model...")
        train_ds = cast(Any, dataloaders["train"].dataset)
        val_ds = cast(Any, dataloaders["validation"].dataset)
        X_tr = train_ds.data["high_level"]
        y_tr = train_ds.data["labels"]
        X_va = val_ds.data["high_level"]
        y_va = val_ds.data["labels"]
        model.fit_from_arrays(X_tr, y_tr, X_va, y_va)
        checkpoint_path = os.path.join(run_dir, "checkpoints", "gbt_model.pkl")
        temporary_checkpoint = checkpoint_path + ".tmp"
        model.save_model(temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint_path)
        return {
            "model": model,
            "regime": regime,
            "backbone": backbone_name,
            "run_dir": run_dir,
            "run_id": run_id,
            "checkpoint_path": checkpoint_path,
            "epoch_metrics_path": None,
            "target_wp": target_wp,
            "seed": seed,
            "trajectories": {},
            "temperature_scaler": None,
        }

    # PyTorch neural backbone training
    loss_fn = build_loss(regime, config, target_wp).to(device)
    reg_cfg = config.get("regime", {})
    opt_cfg = reg_cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 1e-3))
    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = int(config.get("experiment", {}).get("epochs", 10))

    # Risk allocator for PI-WPCT
    risk_allocator = None
    refresh_interval_k = 2
    if regime == "pi_wpct":
        ra_cfg = reg_cfg.get("risk_allocator", {})
        refresh_interval_k = int(ra_cfg.get("refresh_interval_k", 2))
        stab_cfg = ra_cfg.get("variance_stabilization", {})
        risk_allocator = RiskAllocator(
            lambda_bg=float(reg_cfg.get("loss", {}).get("lambda_bg", 1.0)),
            variance_stabilization=stab_cfg.get("method", "wilson_floor"),
            min_variance_floor=float(stab_cfg.get("min_variance_floor", 1e-7)),
            z_score=float(stab_cfg.get("z_score", 1.96)),
            zero_sum_guard_epsilon=float(ra_cfg.get("zero_sum_guard_epsilon", 1e-12)),
            fixed_yields=config.get("data", {}).get("yields", None),
        )

    # Initial risk refresh at epoch 0 if PI-WPCT
    if regime == "pi_wpct" and risk_allocator is not None:
        logger.info("Performing pre-training (epoch 0) risk allocation refresh on validation split...")
        res = risk_allocator.refresh(model, dataloaders["validation"], target_wp, device)
        cast(PIWPCTLoss, loss_fn).update_weights(res.lambda_light, res.lambda_charm)
        trajectory_logger.log_refresh(
            epoch=0,
            target_wp=target_wp,
            r_light=res.r_light,
            r_charm=res.r_charm,
            var_r_light=res.var_r_light,
            var_r_charm=res.var_r_charm,
            purity=res.purity,
            dp_dr_light=res.dp_dr_light,
            dp_dr_charm=res.dp_dr_charm,
            r_light_risk=res.r_light_risk,
            r_charm_risk=res.r_charm_risk,
            lambda_light=res.lambda_light,
            lambda_charm=res.lambda_charm,
            extra={"action": "initial_refresh"},
        )
        logger.info(
            f"Epoch 0 weights: lambda_light={res.lambda_light:.4f}, lambda_charm={res.lambda_charm:.4f}, "
            f"purity={res.purity:.4f}, risk_ratio={res.risk_ratio:.4f}"
        )

    best_val_loss = float("inf")
    best_epoch = 0
    best_weights = copy.deepcopy(model.state_dict())
    epoch_metrics_logger = EpochMetricsLogger(
        os.path.join(run_dir, "metrics", "epochs.jsonl"), run_id=run_id
    )

    # Training loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, dataloaders["train"], optimizer, loss_fn, device, target_wp)
        val_metrics, val_logits, val_labels = evaluate_epoch(
            model, dataloaders["validation"], loss_fn, device, target_wp,
            collect_outputs=False,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch}/{epochs} ({elapsed:.1f}s) - "
            f"Train Loss: {train_metrics['train_loss']:.4f} - Val Loss: {val_metrics['val_loss']:.4f}"
        )
        epoch_metrics_logger.log_epoch(epoch, train_metrics, val_metrics)

        # Periodic risk refresh hook for PI-WPCT at K-epoch boundaries
        if regime == "pi_wpct" and risk_allocator is not None and (epoch % refresh_interval_k == 0):
            res = risk_allocator.refresh(model, dataloaders["validation"], target_wp, device)
            cast(PIWPCTLoss, loss_fn).update_weights(res.lambda_light, res.lambda_charm)
            trajectory_logger.log_refresh(
                epoch=epoch,
                target_wp=target_wp,
                r_light=res.r_light,
                r_charm=res.r_charm,
                var_r_light=res.var_r_light,
                var_r_charm=res.var_r_charm,
                purity=res.purity,
                dp_dr_light=res.dp_dr_light,
                dp_dr_charm=res.dp_dr_charm,
                r_light_risk=res.r_light_risk,
                r_charm_risk=res.r_charm_risk,
                lambda_light=res.lambda_light,
                lambda_charm=res.lambda_charm,
                extra={"val_loss": val_metrics["val_loss"]},
            )
            logger.info(
                f"[Risk Refresh @ Epoch {epoch}] lambda_light={res.lambda_light:.4f}, "
                f"lambda_charm={res.lambda_charm:.4f}, r_l={res.r_light:.4f}, r_c={res.r_charm:.4f}, "
                f"purity={res.purity:.4f}"
            )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_weights)

    # Post-hoc temperature scaling if applicable
    temp_scaler = None
    if regime == "ce_temp_scaling":
        logger.info("Fitting post-hoc temperature scaling on validation logits...")
        _, val_logits, val_labels = evaluate_epoch(model, dataloaders["validation"], loss_fn, device, target_wp)
        temp_scaler = TemperatureScaler()
        fitted_t = temp_scaler.fit(val_logits, val_labels)
        logger.info(f"Optimal temperature parameter: T={fitted_t:.4f}")

    # Save checkpoint with rich metadata
    checkpoint_path = os.path.join(run_dir, "checkpoints", "checkpoint.pt")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    temporary_checkpoint = checkpoint_path + ".tmp"
    manifest_hash = "unknown"
    dataset_auditor = getattr(train_dataset, "auditor", None)
    if dataset_auditor and dataset_auditor.manifest_path:
        manifest_hash = SplitAuditor.compute_sha256(dataset_auditor.manifest_path)
    elif hasattr(train_dataset, "manifest_path"):
        manifest_hash = SplitAuditor.compute_sha256(str(train_dataset.manifest_path))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "regime": regime,
            "backbone": backbone_name,
            "target_wp": target_wp,
            "seed": seed,
            "epochs": epochs,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "run_id": run_id,
            "git_commit": get_git_commit_hash(),
            "manifest_hash": manifest_hash,
            "target_mode": target_mode,
            "class_names": class_names,
            "config": config,
            "trajectories": trajectory_logger.get_trajectories(),
            "temperature": float(temp_scaler.temperature.item()) if temp_scaler else 1.0,
        },
        temporary_checkpoint,
    )
    os.replace(temporary_checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

    return {
        "model": model,
        "regime": regime,
        "backbone": backbone_name,
        "run_dir": run_dir,
        "run_id": run_id,
        "target_wp": target_wp,
        "seed": seed,
        "trajectories": trajectory_logger.get_trajectories(),
        "temperature_scaler": temp_scaler,
        "checkpoint_path": checkpoint_path,
        "epoch_metrics_path": os.path.join(run_dir, "metrics", "epochs.jsonl"),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def train_pipeline(
    regime: str,
    backbone_name: str,
    config: Dict[str, Any],
    dataloaders: Dict[str, DataLoader],
    target_wp: float = 0.70,
    seed: int = 42,
    device_str: Optional[str] = None,
    output_dir: str = "outputs",
    run_id: Optional[str] = None,
    run_parent_dir: Optional[str] = None,
    run_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Reserve a unique run directory, then execute the unchanged training pipeline."""
    train_dataset = dataloaders["train"].dataset
    target_mode = getattr(train_dataset, "target_mode", "flavour_tagging")
    dataset_auditor = getattr(train_dataset, "auditor", None)
    manifest_path = getattr(train_dataset, "manifest_path", None)
    if manifest_path is None and dataset_auditor is not None:
        manifest_path = getattr(dataset_auditor, "manifest_path", None)
    dataset_mode = "real" if hasattr(train_dataset, "manifest_path") else "pilot" if dataset_auditor else "unknown"
    identity: Dict[str, Any] = {
        "run_type": "training",
        "regime": regime,
        "backbone": backbone_name,
        "target_wp": float(target_wp),
        "seed": int(seed),
        "dataset_mode": dataset_mode,
        "target_mode": target_mode,
        "effective_config": config,
        "effective_training_config": {
            "regime": regime,
            "backbone": backbone_name,
            "target_wp": float(target_wp),
            "seed": int(seed),
            "epochs": int(config.get("experiment", {}).get("epochs", 10)),
        },
        **manifest_identifier(str(manifest_path) if manifest_path else None),
    }
    if run_metadata:
        identity.update(run_metadata)
    run_dir, selected_run_id = create_run_directory(
        output_dir,
        identity=identity,
        run_id=run_id,
        run_parent_dir=run_parent_dir,
    )
    try:
        result = _train_pipeline_in_run(
            regime=regime,
            backbone_name=backbone_name,
            config=config,
            dataloaders=dataloaders,
            target_wp=target_wp,
            seed=seed,
            device_str=device_str,
            output_dir=str(run_dir),
            run_id=selected_run_id,
        )
    except BaseException as exc:
        update_run_metadata(
            str(run_dir),
            status="failed",
            failure={"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    update_run_metadata(
        str(run_dir),
        status="training_complete",
        checkpoint_path=result.get("checkpoint_path"),
        trajectories_path=str(run_dir / "trajectories" / "trajectories.jsonl"),
        best_epoch=result.get("best_epoch"),
        best_val_loss=result.get("best_val_loss"),
        paths={
            "checkpoint": result.get("checkpoint_path"),
            "epoch_metrics": result.get("epoch_metrics_path"),
        },
    )
    result["run_dir"] = str(run_dir)
    result["run_id"] = selected_run_id
    return result
