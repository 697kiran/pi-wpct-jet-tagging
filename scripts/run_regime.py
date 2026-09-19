"""Execute a single training regime and evaluation run.

Usage:
    python scripts/run_regime.py --regime pi_wpct --backbone pn_lite --wp 0.70 --seed 42 --epochs 5
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from piwpct.conformal import MondrianConformalCalibrator
from piwpct.data import get_dataloaders
from piwpct.evaluation.plots import plot_conformal_coverage, plot_lambda_trajectories, plot_robustness_curves
from piwpct.evaluation.robustness import evaluate_robustness_grid
from piwpct.metrics import compute_classification_metrics, compute_ece, compute_working_point_metrics
from piwpct.train import train_pipeline
from piwpct.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run training and evaluation for a specific regime.")
    parser.add_argument("--regime", type=str, default="pi_wpct", choices=[
        "baseline_ce", "ce_temp_scaling", "ce_mondrian_cp", "conftr", "wpct", "pi_wpct"
    ])
    parser.add_argument("--backbone", type=str, default="pn_lite", choices=[
        "pn_lite", "lorentznet_small", "part", "gbt"
    ])
    parser.add_argument("--wp", type=float, default=0.70, help="Target b-efficiency working point")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--split-manifest", type=str, default="data/splits/canonical_split_manifest.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    # Load configurations
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base_dir, "configs", "experiment.yaml")) as f:
        exp_cfg = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "data.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "regimes", f"{args.regime}.yaml")) as f:
        reg_cfg = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "backbones", f"{args.backbone}.yaml")) as f:
        bb_cfg = yaml.safe_load(f)

    full_config = {
        "experiment": exp_cfg.get("experiment", {}),
        "data": data_cfg.get("data", {}),
        "regime": reg_cfg,
        "backbone": bb_cfg,
    }
    if args.epochs is not None:
        full_config["experiment"]["epochs"] = args.epochs

    seed_everything(args.seed)

    # Dataloaders
    dataloaders = get_dataloaders(args.split_manifest, batch_size=256)

    # 1. Execute shared training loop
    train_results = train_pipeline(
        regime=args.regime,
        backbone_name=args.backbone,
        config=full_config,
        dataloaders=dataloaders,
        target_wp=args.wp,
        seed=args.seed,
        output_dir=args.output_dir,
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
                l = model(features=f, points=p, mask=m, vectors=v, high_level=hl).cpu().numpy()
            all_l.append(l)
            all_y.append(batch["labels"].numpy())
        return np.concatenate(all_l, axis=0), np.concatenate(all_y, axis=0)

    # 2. Exact Mondrian Conformal Calibration on CALIBRATION split
    print("[*] Running exact Mondrian conformal calibration on frozen calibration split...")
    cal_logits, cal_labels = _get_logits(dataloaders["calibration"])
    calibrator = MondrianConformalCalibrator(alpha=0.10, score_type="residual")
    calibrator.calibrate(cal_logits, cal_labels)

    # 3. Final evaluation on TEST split
    print("[*] Evaluating on held-out test split...")
    test_logits, test_labels = _get_logits(dataloaders["test"])

    # If temperature scaling regime, apply fitted temperature
    if train_results.get("temperature_scaler"):
        t_val = float(train_results["temperature_scaler"].temperature.item())
        test_logits = test_logits / t_val

    wp_metrics = compute_working_point_metrics(test_logits, test_labels, target_wp=args.wp)
    cls_metrics = compute_classification_metrics(test_logits, test_labels)
    ece_val = compute_ece(cls_metrics["probabilities"], test_labels)
    conformal_out = calibrator.evaluate(test_logits, test_labels)

    print("\n" + "=" * 55)
    print(f"   FINAL TEST RESULTS: {args.regime.upper()} ({args.backbone})")
    print("=" * 55)
    print(f"Accuracy:                   {cls_metrics['accuracy']*100:.2f}%")
    print(f"Macro-AUC:                  {cls_metrics['macro_auc']:.4f}")
    print(f"ECE:                        {ece_val:.4f}")
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
    plot_conformal_coverage(conformal_out.per_class_coverage, target_coverage=0.90, save_path=cov_plot)

    # 4. Detector-mismodelling robustness tests on test data
    print("[*] Running detector-mismodelling robustness perturbations...")
    test_ds = dataloaders["test"].dataset
    robustness_data = evaluate_robustness_grid(
        model=model,
        calibrator=calibrator,
        test_data=test_ds.data,
        target_wp=args.wp,
        device=device,
    )
    rob_plot = os.path.join(run_dir, "robustness_curves.png")
    plot_robustness_curves(robustness_data, save_path=rob_plot)
    print(f"[+] Robustness analysis complete. Plots saved to: {rob_plot}")


if __name__ == "__main__":
    main()
