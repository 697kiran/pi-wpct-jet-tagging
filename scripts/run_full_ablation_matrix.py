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
from piwpct.data_paths import PROJECT_ROOT, load_data_config, resolve_data_root, resolve_project_path
from piwpct.evaluation.report import EvaluationReporter
from piwpct.metrics import compute_classification_metrics, compute_ece, compute_working_point_metrics, paired_bootstrap_comparison
from piwpct.train import scale_calibration_and_test_logits, train_pipeline
from piwpct.utils.run_directory import (
    create_run_directory,
    manifest_identifier,
    update_run_metadata,
    write_run_json,
)
from piwpct.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run full PI-WPCT ablation matrix.")
    add_data_mode_arguments(parser)
    parser.add_argument("--backbone", type=str, default="pn_lite", help="Backbone model to benchmark")
    parser.add_argument("--regimes", nargs="+", default=["baseline_ce", "wpct", "pi_wpct"], help="Regimes to benchmark")
    parser.add_argument("--wps", nargs="+", type=float, default=[0.70], help="Working points to benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds to evaluate")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch budget per run")
    parser.add_argument("--split-manifest", type=str, default=None)
    parser.add_argument("--prepared-data-root", type=str, default=None)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", default=None)
    parser.add_argument("--run-id", default=None, help="Optional explicit ablation-suite ID; collisions are refused")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--particle-cap", type=int, default=None)
    parser.add_argument("--target-mode", choices=("process_classification", "flavour_tagging"),
                        help="Required with --real-data; this ablation uses flavour-specific losses")
    parser.add_argument("--allow-incomplete-data", action="store_true",
                        help="Allow a bounded smoke-conversion manifest for testing only")
    args = parser.parse_args()

    if args.data_mode == "real" and args.target_mode is None:
        parser.error("--real-data requires an explicit --target-mode matching the prepared manifest.")

    base_dir = str(PROJECT_ROOT)
    data_cfg = load_data_config(os.path.join(base_dir, "configs", "data.yaml"))
    with open(os.path.join(base_dir, "configs", "backbones", f"{args.backbone}.yaml")) as f:
        bb_cfg = yaml.safe_load(f)

    prepared_root, _ = resolve_data_root("prepared_data_root", args.prepared_data_root, data_cfg)
    output_root, _ = resolve_data_root("output_root", args.output_root, data_cfg)
    manifest_path = str(resolve_project_path(args.split_manifest)) if args.split_manifest else (
        str(prepared_root / "canonical_split_manifest.json")
        if args.data_mode == "real" else str(resolve_project_path("data/splits/canonical_split_manifest.json"))
    )
    try:
        manifest = ensure_split_manifest(
            manifest_path,
            data_mode=args.data_mode,
            generate_pilot_if_missing=(args.data_mode == "pilot"),
            seed=args.seeds[0],
        )
    except SplitManifestError as exc:
        parser.error(str(exc))

    target_mode = manifest.get("target_definition", {}).get("mode", "flavour_tagging")
    if args.data_mode == "real" and target_mode != args.target_mode:
        parser.error(f"Requested target mode {args.target_mode!r} does not match manifest target {target_mode!r}.")
    if target_mode != "flavour_tagging":
        parser.error("This ablation includes WPCT/PI-WPCT and requires valid light/charm/b labels.")
    class_names = manifest.get("target_definition", {}).get("class_names", ["light", "charm", "b"])
    bb_cfg["num_classes"] = len(class_names)
    bb_cfg["input_dim"] = len(manifest.get("feature_schema", {}).get("ordering", [])) or 17
    bb_cfg["n_scalar"] = len(manifest.get("feature_schema", {}).get("ordering", [])) or 17

    loader_cfg = data_cfg.get("dataloader", {})
    workers = args.workers if args.workers is not None else int(loader_cfg.get("num_workers", 0))
    dataloaders = get_dataloaders(
        manifest_path,
        batch_size=int(loader_cfg.get("batch_size", 256)),
        num_workers=workers,
        pin_memory=args.pin_memory or bool(loader_cfg.get("pin_memory", False)),
        shuffle_buffer_size=int(loader_cfg.get("shuffle_buffer_size", 8192)),
        particle_cap=args.particle_cap,
        allow_incomplete=args.allow_incomplete_data,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    suite_dir, suite_id = create_run_directory(
        str(output_root),
        run_id=args.run_id,
        identity={
            "run_type": "ablation_suite",
            "dataset_mode": args.data_mode,
            "dataset_identifier": manifest.get("dataset_name", Path(manifest_path).stem),
            "target_mode": target_mode,
            "backbone": args.backbone,
            "requested_regimes": list(args.regimes),
            "requested_working_points": list(args.wps),
            "requested_seeds": list(args.seeds),
            "effective_run_selection": {
                "regimes": list(args.regimes),
                "working_point": args.wps[0],
                "seed": args.seeds[0],
            },
            "effective_config": {"epochs": args.epochs, "data": data_cfg, "backbone": bb_cfg},
            **manifest_identifier(manifest_path),
        },
    )
    child_run_directories = []

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
                batch_logits = model(features=f, points=p, mask=m, vectors=v, high_level=hl).cpu().numpy()
            all_l.append(batch_logits)
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
            "data": data_cfg,
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
            output_dir=str(output_root),
            run_parent_dir=str(suite_dir / "runs"),
            run_metadata={
                "run_type": "ablation_child",
                "dataset_mode": args.data_mode,
                "dataset_identifier": manifest.get("dataset_name", Path(manifest_path).stem),
                "target_mode": target_mode,
            },
        )
        child_run_directories.append(train_res["run_dir"])

        model = train_res["model"]

        # Exact Mondrian Conformal Calibration on held-out CALIBRATION split
        cal_logits, cal_labels = _get_logits(model, dataloaders["calibration"])
        # Final Evaluation on held-out TEST split
        test_logits, test_labels = _get_logits(model, dataloaders["test"])
        temperature = 1.0
        if train_res.get("temperature_scaler"):
            temperature = float(train_res["temperature_scaler"].temperature.item())
        cal_logits, test_logits = scale_calibration_and_test_logits(cal_logits, test_logits, temperature)
        calibrator = MondrianConformalCalibrator(alpha=0.10, score_type="residual")
        calibrator.calibrate(cal_logits, cal_labels)
        test_labels_ref = test_labels
        test_logits_by_regime[reg] = test_logits

        wp_m = compute_working_point_metrics(test_logits, test_labels, target_wp=wp)
        cls_m = compute_classification_metrics(test_logits, test_labels)
        ece_val = compute_ece(cls_m["probabilities"], test_labels)
        conf_m = calibrator.evaluate_batches([(test_logits, test_labels)])

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
        child_metrics_path = write_run_json(
            train_res["run_dir"],
            "metrics/final_metrics.json",
            {
                "schema": "piwpct.final_metrics.v1",
                "run_id": train_res["run_id"],
                "dataset_identifier": manifest.get("dataset_name", Path(manifest_path).stem),
                "dataset_mode": args.data_mode,
                "target_mode": target_mode,
                "regime": reg,
                "backbone": args.backbone,
                "target_wp": wp,
                "seed": seed,
                "metrics": regime_results[reg],
            },
        )
        calibration_path = write_run_json(
            train_res["run_dir"],
            "metrics/conformal_calibration.json",
            {
                "schema": "piwpct.conformal_calibration.v1",
                "run_id": train_res["run_id"],
                "method": "mondrian",
                "score_type": calibrator.score_type,
                "alpha": calibrator.alpha,
                "class_quantiles": {str(key): float(value) for key, value in calibrator.class_quantiles.items()},
                "calibration_count_by_class": {
                    str(index): int(np.sum(cal_labels == index)) for index in range(len(class_names))
                },
                "temperature": temperature,
            },
        )
        update_run_metadata(
            train_res["run_dir"],
            status="completed",
            final_metrics_path=str(child_metrics_path),
            conformal_calibration_path=str(calibration_path),
        )

    # Paired Bootstrap Comparison between WPCT and PI-WPCT
    paired_comparisons = None
    if "wpct" in test_logits_by_regime and "pi_wpct" in test_logits_by_regime:
        print("\n[*] Performing paired bootstrap comparison between PI-WPCT and WPCT (1000 resamples)...")
        target_wp = args.wps[0]
        def _metric_fn(logits, labels):
            return compute_working_point_metrics(logits, labels, target_wp=target_wp)

        paired_comparisons = paired_bootstrap_comparison(
            logits_a=test_logits_by_regime["wpct"],
            logits_b=test_logits_by_regime["pi_wpct"],
            labels=test_labels_ref,
            metric_fn=_metric_fn,
            n_bootstraps=1000,
            seed=42,
        )

    # Generate final report with automated failure-mode / null-result checks
    reporter = EvaluationReporter(output_dir=str(suite_dir / "reports"))
    report_file = reporter.generate_markdown_report(
        experiment_name=f"Ablation Matrix ({args.backbone}, WP={args.wps[0]})",
        regime_results=regime_results,
        paired_comparisons=paired_comparisons,
        output_filename="ablation_summary_report.md",
    )
    print("\n[+] Full ablation matrix finished successfully!")
    print(f"[+] Final report generated at: {report_file}")
    report_json = str(Path(report_file).with_suffix(".json"))
    update_run_metadata(
        str(suite_dir),
        status="completed",
        child_run_directories=child_run_directories,
        report_markdown=report_file,
        report_json=report_json,
    )


if __name__ == "__main__":
    main()
