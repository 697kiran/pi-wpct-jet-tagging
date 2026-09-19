"""Mandatory pre-training risk-curve diagnostic script (§6 & §9.2).

Computes R_light(w), R_charm(w), and their ratio across all candidate working points
(e.g., 50%, 70%, 77%, 85%) using a pilot model or existing checkpoint on the validation split.
Determines whether there is sufficient WP-dependent risk variation to justify PI-WPCT,
and alerts if the risk ratio is flat (failure/negative result criterion §9.3).

Usage:
    python scripts/run_pretraining_diagnostic.py --config configs/experiment.yaml
"""

import argparse
import os
import sys
import yaml
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from piwpct.backbones import build_backbone
from piwpct.data import get_dataloaders
from piwpct.evaluation.plots import plot_risk_curves
from piwpct.risk_allocator import RiskAllocator
from piwpct.utils.seeding import seed_everything


def main():
    parser = argparse.ArgumentParser(description="Run mandatory pre-training PI-WPCT diagnostic.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Path to experiment config")
    parser.add_argument("--split-manifest", type=str, default="data/splits/canonical_split_manifest.json", help="Manifest path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional model checkpoint to evaluate")
    parser.add_argument("--output-plot", type=str, default="plots/pretraining_risk_diagnostic.png", help="Path to save plot")
    parser.add_argument("--pilot", action="store_true", default=False, help="Generate pilot split if missing")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed_everything(config.get("experiment", {}).get("default_seed", 42))

    # Check if manifest exists, if not and pilot flag is given, create it
    if not os.path.exists(args.split_manifest):
        print(f"[!] Split manifest '{args.split_manifest}' not found. Building pilot split...")
        from piwpct.data import create_canonical_splits, generate_pilot_dataset
        data = generate_pilot_dataset(n_samples=12000, seed=42)
        os.makedirs(os.path.dirname(args.split_manifest), exist_ok=True)
        create_canonical_splits(data, output_dir=os.path.dirname(args.split_manifest), seed=42, is_pilot=True)

    dataloaders = get_dataloaders(args.split_manifest, batch_size=256)
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

    print(f"\n=================================================================")
    print(f"   MANDATORY PRE-TRAINING PI-WPCT RISK DIAGNOSTIC (§6 & §9.2)     ")
    print(f"=================================================================")
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
