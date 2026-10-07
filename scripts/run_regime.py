"""Execute a single training regime and evaluation run.

Usage:
    python scripts/run_regime.py --pilot --regime pi_wpct --backbone pn_lite --wp 0.70 --seed 42 --epochs 5
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from piwpct.conformal import MondrianConformalCalibrator
from piwpct.cli_data import (
    SplitManifestError,
    add_data_mode_arguments,
    ensure_split_manifest,
)
from piwpct.data import get_dataloaders
from piwpct.data_paths import load_data_config, resolve_data_root
from piwpct.evaluation.plots import plot_conformal_coverage, plot_lambda_trajectories, plot_robustness_curves
from piwpct.evaluation.robustness import evaluate_robustness_grid, evaluate_robustness_loader
from piwpct.metrics import compute_classification_metrics, compute_ece, compute_working_point_metrics
from piwpct.train import scale_calibration_and_test_logits, train_pipeline
from piwpct.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run training and evaluation for a specific regime.")
    add_data_mode_arguments(parser)
    parser.add_argument("--regime", type=str, default="pi_wpct", choices=[
        "baseline_ce", "ce_temp_scaling", "ce_mondrian_cp", "conftr", "wpct", "pi_wpct"
    ])
    parser.add_argument("--backbone", type=str, default="pn_lite", choices=[
        "pn_lite", "lorentznet_small", "part", "gbt"
    ])
    parser.add_argument("--wp", type=float, default=0.70, help="Target b-efficiency working point")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--split-manifest", type=str, default=None, help="Prepared/pilot split manifest path")
    parser.add_argument("--prepared-data-root", type=str, default=None, help="Prepared shard root (real-data mode)")
    parser.add_argument("--output-root", "--output-dir", dest="output_root", default=None,
                        help="Root directory for checkpoints, logs and reports")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers (configurable; default 0)")
    parser.add_argument("--pin-memory", action="store_true", help="Pin batches for accelerator transfer")
    parser.add_argument("--particle-cap", type=int, default=None, help="Batch-time particle cap (default 128)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides data config)")
    parser.add_argument("--target-mode", choices=("process_classification", "flavour_tagging"),
                        help="Required with --real-data and checked against the prepared manifest")
    parser.add_argument("--allow-incomplete-data", action="store_true",
                        help="Allow a bounded smoke-conversion manifest for a tiny pipeline check")
    args = parser.parse_args()

    if args.data_mode == "real" and args.target_mode is None:
        parser.error("--real-data requires an explicit --target-mode matching the prepared manifest.")

    # Load configurations
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base_dir, "configs", "experiment.yaml")) as f:
        exp_cfg = yaml.safe_load(f)
    data_cfg = load_data_config(os.path.join(base_dir, "configs", "data.yaml"))
    with open(os.path.join(base_dir, "configs", "regimes", f"{args.regime}.yaml")) as f:
        reg_cfg = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "backbones", f"{args.backbone}.yaml")) as f:
        bb_cfg = yaml.safe_load(f)

    full_config = {
        "experiment": exp_cfg.get("experiment", {}),
        "data": data_cfg,
        "regime": reg_cfg,
        "backbone": bb_cfg,
    }
    if args.epochs is not None:
        full_config["experiment"]["epochs"] = args.epochs

    seed_everything(args.seed)

    # Dataloaders
    prepared_root, _ = resolve_data_root("prepared_data_root", args.prepared_data_root, data_cfg)
    output_root, _ = resolve_data_root("output_root", args.output_root, data_cfg)
    manifest_path = args.split_manifest or (
        str(Path(prepared_root) / "canonical_split_manifest.json")
        if args.data_mode == "real"
        else "data/splits/canonical_split_manifest.json"
    )
    try:
        manifest = ensure_split_manifest(manifest_path, data_mode=args.data_mode)
    except SplitManifestError as exc:
        parser.error(str(exc))
    target_definition = manifest.get("target_definition", {})
    target_mode = target_definition.get("mode", "flavour_tagging")
    if args.data_mode == "real" and target_mode != args.target_mode:
        parser.error(
            f"Requested target mode {args.target_mode!r} does not match manifest target {target_mode!r}."
        )
    if target_mode == "process_classification" and args.regime in {"wpct", "pi_wpct"}:
        parser.error("WPCT and PI-WPCT require light/charm/b flavour labels; process classes have different semantics.")
    class_names = list(target_definition.get("class_names", ["light", "charm", "b"]))
    bb_cfg["num_classes"] = len(class_names)
    bb_cfg["input_dim"] = len(manifest.get("feature_schema", {}).get("ordering", [])) or bb_cfg.get("input_dim", 17)
    bb_cfg["n_scalar"] = len(manifest.get("feature_schema", {}).get("ordering", [])) or bb_cfg.get("n_scalar", 17)
    dataloader_cfg = data_cfg.get("dataloader", {})
    batch_size = args.batch_size or int(dataloader_cfg.get("batch_size", 256))
    workers = args.workers if args.workers is not None else int(dataloader_cfg.get("num_workers", 0))
    pin_memory = args.pin_memory or bool(dataloader_cfg.get("pin_memory", False))
    dataloaders = get_dataloaders(
        manifest_path,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=pin_memory,
        shuffle_buffer_size=int(dataloader_cfg.get("shuffle_buffer_size", 8192)),
        seed=args.seed,
        particle_cap=args.particle_cap,
        drop_last=bool(dataloader_cfg.get("drop_last", False)),
        allow_incomplete=args.allow_incomplete_data,
    )

    # 1. Execute shared training loop
    train_results = train_pipeline(
        regime=args.regime,
        backbone_name=args.backbone,
        config=full_config,
        dataloaders=dataloaders,
        target_wp=args.wp,
        seed=args.seed,
        output_dir=str(output_root),
    )

    model = train_results["model"]
    run_dir = train_results["run_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Plot lambda trajectories if PI-WPCT
    if args.regime == "pi_wpct" and train_results.get("trajectories"):
        traj_plot = os.path.join(run_dir, "lambda_trajectories.png")
        plot_lambda_trajectories(train_results["trajectories"], traj_plot)

    # Helper function to get model logits on a dataset
    def _get_logits(loader):
        model.eval()
        all_l, all_y = [], []
        for batch in loader:
            f = batch["features"].to(device)
            p = batch["points"].to(device) if "points" in batch else None
            m = batch["mask"].to(device) if "mask" in batch else None
            v = batch["vectors"].to(device) if "vectors" in batch else None
            hl = batch["high_level"].to(device) if "high_level" in batch else None
            with torch.no_grad():
                batch_logits = model(features=f, points=p, mask=m, vectors=v, high_level=hl).cpu().numpy()
            all_l.append(batch_logits)
            all_y.append(batch["labels"].numpy())
        return np.concatenate(all_l, axis=0), np.concatenate(all_y, axis=0)

    # 2. Exact Mondrian Conformal Calibration on CALIBRATION split
    print("[*] Running exact Mondrian conformal calibration on frozen calibration split...")
    cal_logits, cal_labels = _get_logits(dataloaders["calibration"])
    # 3. Final evaluation on TEST split
    print("[*] Evaluating on held-out test split...")
    test_logits, test_labels = _get_logits(dataloaders["test"])
    temperature = 1.0
    if train_results.get("temperature_scaler"):
        temperature = float(train_results["temperature_scaler"].temperature.item())
    cal_logits, test_logits = scale_calibration_and_test_logits(cal_logits, test_logits, temperature)
    calibrator = MondrianConformalCalibrator(alpha=0.10, score_type="residual")
    calibrator.calibrate(cal_logits, cal_labels)

    cls_metrics = compute_classification_metrics(test_logits, test_labels)
    ece_val = compute_ece(cls_metrics["probabilities"], test_labels)
    conformal_out = calibrator.evaluate_batches([(test_logits, test_labels)])

    print("\n" + "=" * 55)
    print(f"   FINAL TEST RESULTS: {args.regime.upper()} ({args.backbone})")
    print("=" * 55)
    print(f"Accuracy:                   {cls_metrics['accuracy']*100:.2f}%")
    print(f"Macro-AUC:                  {cls_metrics['macro_auc']:.4f}")
    print(f"ECE:                        {ece_val:.4f}")
    print(f"Target definition:          {target_mode} ({', '.join(class_names)})")
    wp_metrics = None
    if target_mode == "flavour_tagging":
        wp_metrics = compute_working_point_metrics(test_logits, test_labels, target_wp=args.wp)
        print(f"Target b-efficiency (w):    {args.wp*100:.1f}%")
        print(f"Empirical b-efficiency:     {wp_metrics['eps_b']*100:.2f}%")
        print(f"Light Mistag Rate:          {wp_metrics['r_light']:.4f}")
        print(f"Charm Mistag Rate:          {wp_metrics['r_charm']:.4f}")
        print(f"Light Rejection (1/mistag): {wp_metrics['rej_light']:.2f}")
        print(f"Charm Rejection (1/mistag): {wp_metrics['rej_charm']:.2f}")
    print(f"Conformal Marginal Cov:     {conformal_out.marginal_coverage*100:.2f}% (target 90%)")
    print(f"Conformal Mean Set Size:    {conformal_out.mean_set_size:.2f}")
    print("=" * 55 + "\n")

    # Plot conformal coverage
    cov_plot = os.path.join(run_dir, "conformal_coverage.png")
    plot_conformal_coverage(
        conformal_out.per_class_coverage,
        target_coverage=0.90,
        class_names=class_names,
        save_path=cov_plot,
    )

    # 4. Detector-mismodelling robustness tests on test data
    if target_mode == "flavour_tagging":
        print("[*] Running detector-mismodelling robustness perturbations...")
        test_ds = dataloaders["test"].dataset
        if hasattr(test_ds, "manifest_path"):
            robustness_data = evaluate_robustness_loader(
                model=model,
                calibrator=calibrator,
                data_loader=dataloaders["test"],
                target_wp=args.wp,
                device=device,
                temperature=temperature,
            )
        else:
            robustness_data = evaluate_robustness_grid(
                model=model,
                calibrator=calibrator,
                test_data=test_ds.data,
                target_wp=args.wp,
                device=device,
                temperature=temperature,
            )
        rob_plot = os.path.join(run_dir, "robustness_curves.png")
        plot_robustness_curves(robustness_data, save_path=rob_plot)
        print(f"[+] Robustness analysis complete. Plots saved to: {rob_plot}")
    else:
        print("[!] Flavour-specific detector robustness metrics are not applied to process classes.")


if __name__ == "__main__":
    main()
