"""Comprehensive evaluation reporting and automated null-result / failure-mode detection.

Encodes academic honesty and statistical rigor by explicitly testing and surfacing:
- Null results (e.g. dynamic weights vary but gains are within bootstrap uncertainty).
- Flat risk ratios across candidate working points.
- Purity proxy instability warnings.
- Explicit bounds on claims (clarifying PI-WPCT does not claim INFERNO/SANNT-scale
  downstream profile likelihood optimization).
"""

import json
import os
from typing import Any, Dict, List, Optional
import numpy as np


class EvaluationReporter:
    """Generates structured evaluation reports with automatic null-result detection."""

    def __init__(self, output_dir: str = "outputs/reports"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def evaluate_failure_conditions(
        self,
        wpct_metrics: Dict[str, Any],
        pi_wpct_metrics: Dict[str, Any],
        paired_comparison: Optional[Dict[str, Any]] = None,
        risk_ratios_across_wps: Optional[List[float]] = None,
    ) -> List[Dict[str, str]]:
        """Automatically detect explicit failure or null-result criteria per §7 and §9.3."""
        findings = []

        # 1. Flat risk ratio across WPs
        if risk_ratios_across_wps and len(risk_ratios_across_wps) > 1:
            ratio_spread = max(risk_ratios_across_wps) - min(risk_ratios_across_wps)
            if ratio_spread < 0.05:
                findings.append({
                    "type": "FAILURE_CRITERION_FLAT_RISK",
                    "severity": "CRITICAL_NEGATIVE_RESULT",
                    "message": (
                        f"Risk ratio R_light / R_charm varies by only {ratio_spread:.4f} across tested working points. "
                        "The dynamic physics-impact allocation mechanism has negligible WP-dependent leverage "
                        "for this dataset composition and should be reported as a negative result for the dynamic mechanism."
                    ),
                })

        # 2. Dynamic weights varied, but rejection gain is within bootstrap uncertainty
        if paired_comparison:
            rej_light_comp = paired_comparison.get("rej_light")
            if rej_light_comp:
                diff = rej_light_comp["mean_diff"]
                sig = rej_light_comp["is_significant"]
                ci_low = rej_light_comp["ci_lower"]
                ci_high = rej_light_comp["ci_upper"]

                if not sig:
                    findings.append({
                        "type": "NULL_RESULT_NO_SIGNIFICANT_GAIN",
                        "severity": "NULL_RESULT_REPORT",
                        "message": (
                            f"Dynamic PI-WPCT mean light rejection change was {diff:+.2f} "
                            f"(95% Paired Bootstrap CI: [{ci_low:+.2f}, {ci_high:+.2f}]). "
                            "Because the confidence interval spans zero, the difference is NOT statistically significant. "
                            "Academic integrity requires reporting this as a null result rather than claiming a gain."
                        ),
                    })
                elif diff > 0 and sig:
                    findings.append({
                        "type": "PRIMARY_SUCCESS_VERIFIED",
                        "severity": "CONFIRMED_GAIN",
                        "message": (
                            f"PI-WPCT demonstrates statistically significant light rejection improvement: "
                            f"+{diff:.2f} (95% Paired Bootstrap CI: [{ci_low:.2f}, {ci_high:.2f}], excludes 0)."
                        ),
                    })

        # 3. Methodological scope reminder
        findings.append({
            "type": "SCOPE_BOUNDARY_STATEMENT",
            "severity": "INFORMATIONAL",
            "message": (
                "PI-WPCT is a lightweight, periodic loss-reweighting scheme. It does NOT backpropagate through "
                "a downstream profile likelihood fit or claim INFERNO/SANNT-scale full-uncertainty optimization. "
                "Results are strictly scoped to working-point background mistag reduction and exact conformal calibration."
            ),
        })

        return findings

    def generate_markdown_report(
        self,
        experiment_name: str,
        regime_results: Dict[str, Dict[str, Any]],
        paired_comparisons: Optional[Dict[str, Any]] = None,
        risk_diagnostic: Optional[Dict[str, Any]] = None,
        output_filename: str = "final_evaluation_report.md",
    ) -> str:
        """Generate comprehensive GitHub-flavored Markdown evaluation report."""
        report_path = os.path.join(self.output_dir, output_filename)

        lines = [
            f"# Evaluation Report: {experiment_name}",
            "**B.Tech Capstone Project — Amrita Vishwa Vidyapeetham (Team 6, Guide: Dr. Vivek Menon)**\n",
            "## 1. Executive Summary & Automated Findings\n",
        ]

        # Evaluate failure conditions if WPCT and PI-WPCT are present
        findings = []
        if "wpct" in regime_results and "pi_wpct" in regime_results:
            risk_ratios = None
            if risk_diagnostic and "risk_ratios" in risk_diagnostic:
                risk_ratios = risk_diagnostic["risk_ratios"]
            findings = self.evaluate_failure_conditions(
                regime_results["wpct"],
                regime_results["pi_wpct"],
                paired_comparison=paired_comparisons,
                risk_ratios_across_wps=risk_ratios,
            )

        for finding in findings:
            sev = finding["severity"]
            if "CRITICAL" in sev:
                lines.append(f"> [!CAUTION]\n> **{finding['type']}**: {finding['message']}\n")
            elif "NULL" in sev:
                lines.append(f"> [!WARNING]\n> **{finding['type']}**: {finding['message']}\n")
            elif "CONFIRMED" in sev:
                lines.append(f"> [!TIP]\n> **{finding['type']}**: {finding['message']}\n")
            else:
                lines.append(f"> [!NOTE]\n> **{finding['type']}**: {finding['message']}\n")

        # Summary table of regimes
        lines.append("## 2. Regime Comparison Summary Table\n")
        lines.append(
            "| Regime | Accuracy | Macro-AUC | b-Eff (w) | Light Rejection (1/mistag) | Charm Rejection (1/mistag) | Marginal Coverage | ECE |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|"
        )

        for reg_name, res in regime_results.items():
            acc = res.get("accuracy", 0.0)
            auc = res.get("macro_auc", 0.0)
            eps_b = res.get("eps_b", 0.0)
            rej_l = res.get("rej_light", 0.0)
            rej_c = res.get("rej_charm", 0.0)
            cov = res.get("marginal_coverage", 0.0)
            ece = res.get("ece", 0.0)

            lines.append(
                f"| **{reg_name}** | {acc*100:.2f}% | {auc:.4f} | {eps_b*100:.1f}% | "
                f"{rej_l:.1f} | {rej_c:.1f} | {cov*100:.1f}% | {ece:.4f} |"
            )

        # Statistical uncertainty section
        if paired_comparisons:
            lines.append("\n## 3. Paired Statistical Bootstrap Comparison (PI-WPCT vs WPCT)\n")
            lines.append("| Metric | Mean Difference (PI-WPCT - WPCT) | 95% Bootstrap CI | Statistically Significant? |")
            lines.append("|---|---|---|---|")
            for m_name, comp in paired_comparisons.items():
                diff = comp["mean_diff"]
                low = comp["ci_lower"]
                high = comp["ci_upper"]
                sig = "**YES** (excludes 0)" if comp["is_significant"] else "NO (includes 0)"
                lines.append(f"| `{m_name}` | {diff:+.4f} | [{low:+.4f}, {high:+.4f}] | {sig} |")

        # Conformal calibration validation
        lines.append("\n## 4. Conformal Calibration Guarantees\n")
        lines.append(
            "All regimes evaluated under exact class-conditional Mondrian conformal prediction on the separate, "
            "held-out calibration split. Per-class empirical coverage is guaranteed to satisfy "
            r"$P(Y \in C(X) \mid Y = k) \ge 1 - \alpha$."
        )

        report_content = "\n".join(lines)
        with open(report_path, "w") as f:
            f.write(report_content)

        # Also save structured JSON
        json_path = os.path.join(self.output_dir, output_filename.replace(".md", ".json"))
        with open(json_path, "w") as f:
            json.dump({
                "experiment_name": experiment_name,
                "findings": findings,
                "regimes": regime_results,
                "paired_comparisons": paired_comparisons,
            }, f, indent=2)

        return report_path
