"""Structured experiment logging and trajectory recording utilities."""

import datetime
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional


def get_git_commit_hash() -> str:
    """Retrieve current git commit hash if in a git repo, else return 'unknown'."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except Exception:
        return "unknown"


def setup_logger(name: str = "piwpct", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Set up formatted console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class TrajectoryLogger:
    """Logs risk refresh events and intermediate optimization variables to JSON Lines."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.events: List[Dict[str, Any]] = []

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a structured event with timestamp."""
        record = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        self.events.append(record)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_refresh(
        self,
        epoch: int,
        target_wp: float,
        r_light: float,
        r_charm: float,
        var_r_light: float,
        var_r_charm: float,
        purity: float,
        dp_dr_light: float,
        dp_dr_charm: float,
        r_light_risk: float,
        r_charm_risk: float,
        lambda_light: float,
        lambda_charm: float,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a full PI-WPCT validation risk refresh event."""
        payload = {
            "epoch": epoch,
            "target_wp": target_wp,
            "r_light": float(r_light),
            "r_charm": float(r_charm),
            "var_r_light": float(var_r_light),
            "var_r_charm": float(var_r_charm),
            "purity": float(purity),
            "dp_dr_light": float(dp_dr_light),
            "dp_dr_charm": float(dp_dr_charm),
            "r_light_risk": float(r_light_risk),
            "r_charm_risk": float(r_charm_risk),
            "risk_ratio": float(r_light_risk / (r_charm_risk + 1e-12)),
            "lambda_light": float(lambda_light),
            "lambda_charm": float(lambda_charm),
        }
        if extra:
            payload.update(extra)
        self.log_event("risk_refresh", payload)

    def get_trajectories(self) -> Dict[str, List[float]]:
        """Extract continuous time series for key quantities."""
        trajectories: Dict[str, List[float]] = {
            "epoch": [],
            "lambda_light": [],
            "lambda_charm": [],
            "r_light": [],
            "r_charm": [],
            "purity": [],
            "r_light_risk": [],
            "r_charm_risk": [],
        }
        for rec in self.events:
            if rec.get("event_type") == "risk_refresh":
                for k in trajectories.keys():
                    if k in rec:
                        trajectories[k].append(rec[k])
        return trajectories
