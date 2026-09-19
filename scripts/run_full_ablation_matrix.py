"""Automate the full ablation matrix across regimes, working points, and seeds.

Compares:
    Baseline CE, CE+TempScale, CE+Mondrian, ConfTr, WPCT, PI-WPCT
Performs paired bootstrap comparisons between WPCT and PI-WPCT,
and generates the final Markdown/JSON evaluation report with failure detection.

Usage:
    python scripts/run_full_ablation_matrix.py --backbone pn_lite --pilot --epochs 3
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from piwpct.conformal import MondrianConformalCalibrator
from piwpct.data import create_canonical_splits, generate_pilot_dataset, get_dataloaders
from piwpct.evaluation.report import EvaluationReporter
from piwpct.metrics import compute_classification_metrics, compute_ece, compute_working_point_metrics, paired_bootstrap_comparison
from piwpct.train import train_pipeline
from piwpct.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run full PI-WPCT ablation matrix.")
    parser.add_argument("--backbone", type=str, default="pn_lite", help="Backbone model to benchmark")
    parser.add_argument("--regimes", nargs="+", default=["baseline_ce", "wpct", "pi_wpct"], help="Regimes to benchmark")
    parser.add_argument("--wps", nargs="+", type=float, default=[0.70], help="Working points to benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch budget per run")
    parser.add_argument("--pilot", action="store_true", default=True, help="Use pilot dataset")
    parser.add_argument("--split-manifest", type=str, default="data/splits/canonical_split_manifest.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    with open(os.path.join(base_dir, "configs", "experiment.yaml")) as f:
        exp_cfg = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "data.yaml")) as f:
        data_cfg = yaml.safe_load(f)
    with open(os.path.join(base_dir, "configs", "backbones", f"{args.backbone}.yaml")) as f:
        bb_cfg = yaml.safe_load(f)

    # Ensure canonical split exists
    if not os.path.exists(args.split_manifest):
        print(f"[*] Generating canonical pilot split in {os.path.dirname(args.split_manifest)}...")
        data = generate_pilot_dataset(n_samples=12000, seed=42)
        os.makedirs(os.path.dirname(args.split_manifest), exist_ok=True)
        create_canonical_splits(data, output_dir=os.path.dirname(args.split_manifest), seed=42, is_pilot=True)

    dataloaders = get_dataloaders(args.split_manifest, batch_size=256)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_logits(model, loader):
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

    regime_results = {}
    test_logits_by_regime = {}
    test_labels_ref = None

    for reg in args.regimes:
        with open(os.path.join(base_dir, "configs", "regimes", f"{reg}.yaml")) as f:
            reg_cfg = yaml.safe_load(f)

        full_config = {
            "experiment": {"epochs": args.epochs},
            "data": data_cfg.get("data", {}),
            "regime": reg_cfg,
            "backbone": bb_cfg,
        }

        wp = args.wps[0]
        seed = args.seeds[0]
        seed_everything(seed)

        print(f"\n[>>>] Running {reg.upper()} ({args.backbone}, WP={wp}, Seed={seed})...")
        train_res = train_pipeline(
            regime=reg,
            backbone_name=args.backbone,
            config=full_config,
            dataloaders=dataloaders,
            target_wp=wp,
            seed=seed,
            output_dir=args.output_dir,
        )

        model = train_res["model"]

        # Exact Mondrian Conformal Calibration on held-out CALIBRATION split
        cal_logits, cal_labels = _get_logits(model, dataloaders["calibration"])
        calibrator = MondrianConformalCalibrator(alpha=0.10, score_type="residual")
        calibrator.calibrate(cal_logits, cal_labels)

        # Final Evaluation on held-out TEST split
        test_logits, test_labels = _get_logits(model, dataloaders["test"])
        test_labels_ref = test_labels
        test_logits_by_regime[reg] = test_logits

        wp_m = compute_working_point_metrics(test_logits, test_labels, target_wp=wp)
        cls_m = compute_classification_metrics(test_logits, test_labels)
        ece_val = compute_ece(cls_m["probabilities"], test_labels)
        conf_m = calibrator.evaluate(test_logits, test_labels)

        regime_results[reg] = {
            "accuracy": cls_m["accuracy"],
            "macro_auc": cls_m["macro_auc"],
            "eps_b": wp_m["eps_b"],
            "r_light": wp_m["r_light"],
            "r_charm": wp_m["r_charm"],
            "rej_light": wp_m["rej_light"],
            "rej_charm": wp_m["rej_charm"],
            "marginal_coverage": conf_m.marginal_coverage,
            "mean_set_size": conf_m.mean_set_size,
            "ece": ece_val,
        }

    # Paired Bootstrap Comparison between WPCT and PI-WPCT
    paired_comparisons = None
    if "wpct" in test_logits_by_regime and "pi_wpct" in test_logits_by_regime:
        print("\n[*] Performing paired bootstrap comparison between PI-WPCT and WPCT (1000 resamples)...")
        target_wp = args.wps[0]
        def _metric_fn(l, y):
            return compute_working_point_metrics(l, y, target_wp=target_wp)

        paired_comparisons = paired_bootstrap_comparison(
            logits_a=test_logits_by_regime["wpct"],
            logits_b=test_logits_by_regime["pi_wpct"],
            labels=test_labels_ref,
            metric_fn=_metric_fn,
            n_bootstraps=1000,
            seed=42,
        )

    # Generate final report with automated failure-mode / null-result checks
    reporter = EvaluationReporter(output_dir=os.path.join(args.output_dir, "reports"))
    report_file = reporter.generate_markdown_report(
        experiment_name=f"Ablation Matrix ({args.backbone}, WP={args.wps[0]})",
        regime_results=regime_results,
        paired_comparisons=paired_comparisons,
        output_filename="ablation_summary_report.md",
    )
    print(f"\n[+] Full ablation matrix finished successfully!")
    print(f"[+] Final report generated at: {report_file}")


if __name__ == "__main__":
    main()
