import sys
import os
import torch
import numpy as np

# Add the src directory so piwpct can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from piwpct.risk_allocator import RiskAllocator
from piwpct.data import generate_pilot_dataset

app = FastAPI(title="PI-WPCT Jet Tagging MVP API")

# Allow CORS for Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    target_wp: float = 0.70

@app.get("/")
def read_root():
    return {"status": "ok", "message": "PI-WPCT Backend Server Running"}

@app.post("/api/simulate-inference")
def simulate_inference(req: InferenceRequest):
    """
    Simulates a forward pass using the PI-WPCT Risk Allocator for the MVP prototype.
    In a fully trained setting, this would load checkpoint.pt and execute model(data).
    """
    try:
        # 1. Generate pilot physics dataset (emulates real JetClass features)
        data = generate_pilot_dataset(n_samples=1000, num_particles=32)
        labels = data["labels"]
        
        # 2. Simulate raw probabilities (logits) from an untrained/random skeleton
        # For a high-fidelity MVP, we introduce a slight separation between classes
        # so the charts in the UI don't just look like pure noise.
        logits = np.random.randn(1000, 3) 
        # Add slight bias to make b-jets look like b-jets, etc.
        logits[labels == 0, 0] += 2.0  # light
        logits[labels == 1, 1] += 2.0  # charm
        logits[labels == 2, 2] += 2.0  # b-jet
        
        # 3. Instantiate the PI-WPCT Risk Allocator
        allocator = RiskAllocator(lambda_bg=1.0)
        
        # 4. Compute Dynamic Risk metrics purely based on this batch
        res = allocator.compute_risk_allocation(
            logits,
            labels,
            target_wp=req.target_wp
        )
        
        return {
            "status": "success",
            "target_wp": res.target_wp,
            "b_efficiency_achieved": res.b_efficiency,
            "purity_proxy": res.purity,
            "metrics": {
                "dynamic_lambda_light": res.lambda_light,
                "dynamic_lambda_charm": res.lambda_charm,
                "mistag_rate_light": res.r_light,
                "mistag_rate_charm": res.r_charm,
                "var_mistag_light": res.var_r_light,
                "var_mistag_charm": res.var_r_charm,
            },
            # Return first 50 points for scatter plot rendering in Next.js
            "scatter_points": [
                {
                    "id": i,
                    "true_class": int(labels[i]),
                    "score_b": float(np.exp(logits[i, 2]) / np.sum(np.exp(logits[i]))), # Softmax to prob
                    "score_light": float(np.exp(logits[i, 0]) / np.sum(np.exp(logits[i]))),
                    "score_charm": float(np.exp(logits[i, 1]) / np.sum(np.exp(logits[i]))),
                } for i in range(50)
            ],
            "debug": {
                "n_b": res.n_b_val,
                "n_light": res.n_light_val,
                "n_charm": res.n_charm_val,
                "threshold_applied": res.threshold
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
