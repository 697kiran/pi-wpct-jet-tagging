"""Mandatory pre-training risk-curve diagnostic script (§6 & §9.2).

Computes R_light(w), R_charm(w), and their ratio across all candidate working points
(e.g., 50%, 70%, 77%, 85%) using a pilot model or existing checkpoint on the validation split.
Determines whether there is sufficient WP-dependent risk variation to justify PI-WPCT,
and alerts if the risk ratio is flat (failure/negative result criterion §9.3).

Usage:
    python scripts/run_pretraining_diagnostic.py --pilot --config configs/experiment.yaml
"""

import argparse
import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from piwpct.backbones import build_backbone
from piwpct.cli_data import (
    SplitManifestError,
    add_data_mode_arguments,
    ensure_split_manifest,
)
from piwpct.data import get_dataloaders
from piwpct.data_paths import load_data_config, resolve_data_root
from piwpct.evaluation.plots import plot_risk_curves
from piwpct.risk_allocator import RiskAllocator
from piwpct.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run mandatory pre-training PI-WPCT diagnostic.")
    add_data_mode_arguments(parser)
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to experiment config")
    parser.add_argument("--split-manifest", type=str, default=None, help="Manifest path")
    parser.add_argument("--prepared-data-root", type=str, default=None, help="Prepared real-data root")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--pin-memory", action="store_true", help="Pin batches for accelerator transfer")
    parser.add_argument("--particle-cap", type=int, default=None, help="Batch-time particle cap")
    parser.add_argument("--target-mode", choices=("process_classification", "flavour_tagging"),
                        help="Required with --real-data; PI-WPCT diagnostics require a flavour target")
    parser.add_argument("--allow-incomplete-data", action="store_true",
                        help="Allow a bounded smoke-conversion manifest for testing only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint to evaluate")
    parser.add_argument("--output-plot", type=str, default="plots/pretraining_risk_diagnostic.png", help="Path to save plot")
    args = parser.parse_args()

    if args.data_mode == "real" and args.target_mode is None:
        parser.error("--real-data requires an explicit --target-mode matching the prepared manifest.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed_everything(config.get("experiment", {}).get("default_seed", 42))

    data_config = load_data_config(os.path.join(os.path.dirname(__file__), "..", "configs", "data.yaml"))
    prepared_root, _ = resolve_data_root("prepared_data_root", args.prepared_data_root, data_config)
    manifest_path = args.split_manifest or (
        str(prepared_root / "canonical_split_manifest.json")
        if args.data_mode == "real" else "data/splits/canonical_split_manifest.json"
    )
    try:
        manifest = ensure_split_manifest(
            manifest_path,
            data_mode=args.data_mode,
            generate_pilot_if_missing=(args.data_mode == "pilot"),
            seed=config.get("experiment", {}).get("default_seed", 42),
        )
    except SplitManifestError as exc:
        parser.error(str(exc))

    target_mode = manifest.get("target_definition", {}).get("mode", "flavour_tagging")
    if args.data_mode == "real" and target_mode != args.target_mode:
        parser.error(f"Requested target mode {args.target_mode!r} does not match manifest target {target_mode!r}.")
    if target_mode != "flavour_tagging":
        parser.error("PI-WPCT risk diagnostics require valid light/charm/b labels; process classes are not flavour tags.")

    dataloader_cfg = data_config.get("dataloader", {})
    workers = args.workers if args.workers is not None else int(dataloader_cfg.get("num_workers", 0))
    dataloaders = get_dataloaders(
        manifest_path,
        batch_size=int(dataloader_cfg.get("batch_size", 256)),
        num_workers=workers,
        pin_memory=args.pin_memory or bool(dataloader_cfg.get("pin_memory", False)),
        shuffle_buffer_size=int(dataloader_cfg.get("shuffle_buffer_size", 8192)),
        particle_cap=args.particle_cap,
        allow_incomplete=args.allow_incomplete_data,
    )
    val_loader = dataloaders["validation"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load or initialize model
    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        backbone_name = ckpt.get("backbone", "pn_lite")
        model = build_backbone(backbone_name, {}).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[+] Loaded model checkpoint from {args.checkpoint}")
    else:
        print("[!] No checkpoint provided; evaluating diagnostic on ParticleNet-Lite baseline initialization...")
        model = build_backbone("pn_lite", {}).to(device)

    allocator = RiskAllocator(lambda_bg=1.0)
    target_wps = config.get("experiment", {}).get("target_wps", [0.50, 0.70, 0.77, 0.85])

    print("\n=================================================================")
    print("   MANDATORY PRE-TRAINING PI-WPCT RISK DIAGNOSTIC (§6 & §9.2)     ")
    print("=================================================================")
    print(f"Evaluating validation split over candidate WPs: {target_wps}\n")

    r_light_risks = []
    r_charm_risks = []
    risk_ratios = []
    lambda_lights = []
    lambda_charms = []

    print(f"{'Target WP':>10} | {'b-Eff':>8} | {'Mistag Light':>12} | {'Mistag Charm':>12} | {'R_light':>10} | {'R_charm':>10} | {'Ratio (L/C)':>11} | {'Lambda_L':>8} | {'Lambda_C':>8}")
    print("-" * 105)

    for wp in target_wps:
        res = allocator.refresh(model, val_loader, target_wp=wp, device=device)
        r_light_risks.append(res.r_light_risk)
        r_charm_risks.append(res.r_charm_risk)
        risk_ratios.append(res.risk_ratio)
        lambda_lights.append(res.lambda_light)
        lambda_charms.append(res.lambda_charm)

        print(
            f"{wp:>10.2f} | {res.b_efficiency:>8.3f} | {res.r_light:>12.4f} | {res.r_charm:>12.4f} | "
            f"{res.r_light_risk:>10.4e} | {res.r_charm_risk:>10.4e} | {res.risk_ratio:>11.4f} | "
            f"{res.lambda_light:>8.4f} | {res.lambda_charm:>8.4f}"
        )

    # Plot risk curves
    plot_risk_curves(target_wps, r_light_risks, r_charm_risks, risk_ratios, save_path=args.output_plot)
    print(f"\n[+] Diagnostic plot saved to: {args.output_plot}")

    # Check failure / null-result criteria per §9.3
    ratio_spread = max(risk_ratios) - min(risk_ratios)
    print(f"\n[Diagnostic Analysis]: Max risk ratio = {max(risk_ratios):.4f}, Min = {min(risk_ratios):.4f}, Spread = {ratio_spread:.4f}")
    if ratio_spread < 0.05:
        print("[!] FAILURE CRITERION TRIGGERED (§9.3):")
        print("    Risk ratio is nearly constant across all WPs. The dynamic mechanism has")
        print("    insufficient leverage on this model/dataset state to justify dynamic reallocation.")
    else:
        print("[+] SUCCESS CRITERION SATISFIED (§6):")
        print("    Sufficient WP-dependent risk variation observed across candidate working points.")
        print("    PI-WPCT loss reweighting has clear mathematical leverage.")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
