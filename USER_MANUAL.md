# PI-WPCT: Conformal Jet Tagging User Manual

Welcome to the Physics-Impact Weighted Working-Point Conformal Training (PI-WPCT) documentation. This user manual is designed for developers, researchers, and engineers onboarding onto the project, providing a line-by-line functional understanding and architectural clarity of the core mathematical components and the frontend deployment.

## Overview of the Architecture

The codebase contains an end-to-end Machine Learning research pipeline connected to a high-fidelity frontend UI. Broadly, the project is structured as follows:

1. **`src/piwpct/`**: The core ML library containing the data loaders, model backbones, mathematical losses (WPCT, PI-WPCT), exact calibration scripts, and risk allocation logic.
2. **`api/`**: A FastAPI backend that serves as an MVP inference abstraction for the frontend, mocking/calling the trained ML model.
3. **`ui/`**: A Next.js (React) front-end applying the Intercom-style (warm cream editorial) design language.
4. **`scripts/`**: Commander scripts to execute training ablation matrices, split building, and regime evaluations.
5. **`configs/`**: Declarative YAML files configuring the experiment, preventing hard-coded parameters in the scripts.

---

## 1. Backend Core Logic (`src/piwpct/`)

We'll dissect the unique mathematical components that define how PI-WPCT integrates dynamic risk into the training loop.

### A. The Risk Allocator (`src/piwpct/risk_allocator.py`)

This file is the crown jewel of the PI-WPCT innovation. It dynamically computes penalty weights ($\lambda_{\text{light}}$ and $\lambda_{\text{charm}}$) without relying on differentiable loss gradients.

* **Line 53: `class RiskAllocator:`**
  This class is instantiated inside the training loop. Its main role is to receive the model, run inference strictly on the validation set, and output fresh weights.
* **Line 82: `compute_variance(self, k, n)`**
  Instead of raw empirical variance (which could be zero and crash the loss), it stabilizes the Mistag variance estimation using either a Wilson floor or absolute minimums. This guarantees mathematical stability at extreme working points.
* **Line 121: `evaluate_validation_logits()`**
  Decorated with `@torch.no_grad()`. The risk allocation *is not part of the backprop graph*. It also validates `split_name` to prevent data leakage from calibration/test splits (`raise SplitIntegrityViolation`).
* **Line 171: `compute_risk_allocation()`**
  The mathematical core. It computes threshold $\tau_w$ based on target b-efficiency, then computes exact validation yields. It uses the analytical derivative function $dP/dr_i$ (sensitivity) multiplied by variance to map the true physical risk $R_i$.
  Finally, lines 259-260 normalize the risks into $\lambda_l$ and $\lambda_c$.

### B. The Loss Functions (`src/piwpct/losses.py`)

This file contains standard cross-entropy and the advanced WPCT losses.

* **Line 30: `class WPCTLoss(nn.Module):`**
  Implements standard Working-Point Conformal Training. Notice the `compute_soft_wp_penalties` function (Line 54). Because sorting and quantiles are non-differentiable, it uses a sigmoidal smooth step function: `torch.sigmoid((b_scores - tau) / self.temperature)`. This creates a soft, differentiable proxy for efficiency and mistag rates that PyTorch can easily backdrop through.
* **Line 131: `class PIWPCTLoss(WPCTLoss):`**
  Inherits everything from standard WPCT but overrides the forward pass. Why? Because the weights `lambda_light` and `lambda_charm` are injected from the outside via `update_weights(self, lambda_light, lambda_charm)`. This isolates the dynamic risk update (validation set) from the gradient update (training set).

### C. The Training Engine (`src/piwpct/train.py`)

This script ties the model, loss, and risk allocator together.

* **Line 185: `train_pipeline(...)`**
  The entry point. It instantiates the model (based on config), loss function, and the optimizer.
* **Line 262: Initial Risk Refresh**
  *Before* training begins (Epoch 0), if using `pi_wpct`, it triggers a baseline `risk_allocator.refresh()`.
* **Line 287: The Epoch Loop**
  It calls `train_epoch()` using strictly the `train` dataloader.
* **Line 301: Periodic Refresh Hook**
  Every $K$ epochs (`epoch % refresh_interval_k == 0`), it halts training, evaluates the frozen model on the validation split, computes the new risks, and updates the weights in the loss function (`loss_fn.update_weights(...)`).

---

## 2. API Integration (`api/main.py`)

*   **Lines 19-25: CORS Middleware**
    Since the Next.js frontend (port 3000) calls the FastAPI backend (port 8000), CORS handles cross-origin policies.
*   **Line 34: `@app.post("/api/simulate-inference")`**
    This endpoint mocks inference for the MVP UI. Instead of loading massive PyTorch weights, it uses `generate_pilot_dataset` to synthesize features, manually shifts the means (`logits[labels == 0, 0] += 2.0`), and actually runs the true `RiskAllocator`. This allows the UI to demonstrate dynamically changing lambdas using the real mathematical logic, just over mocked synthetic distributions.

---

## 3. High-Fidelity Next.js Frontend (`ui/src/app/page.tsx`)

The UI codebase takes an editorial, heavily designed approach for showcasing the telemetry. 

*   **Lines 3-7: The Imports**
    Standard React tools and `Recharts` for charting. Recharts is chosen for declarative graph generation which fits well into React tree reconciliation.
*   **Lines 40-55: Application State Hooks**
    The main component `Dashboard` utilizes state to track `workingPoint` (default `0.75`), `loading`, and the response `metrics` containing the calculated Lambdas and Mistag rates. Note the `scatterData` array holding `ScatterPoint` records.
*   **Lines 57-93: `runSimulation` Callback**
    This function explicitly fetches from the FastAPI backend at `http://localhost:8000/api/simulate-inference`. It passes the physical `target_wp`. Once it receives the payload, it destructures the response into `metrics` (dynamic lambdas/rates) and maps `data.scatter_points` to visual coordinates based on semantic class labels.
*   **Lines 95-97: The Slider Handle**
    The `handleSliderMouseUp` ensures we only simulate when the user releases the slider, preventing API overload if the user rapidly scrubs the bar.
*   **Lines 238-248 (Working Point Slider):**
    Allows dynamic sweeping of the parameter $w \in [0.50, 0.99]$. This is critical because WPCT aims to dynamically tune to the chosen $w$. 
*   **Lines 362-392 (Scatter Plot component):**
    Visualizes the model's Softmax geometry mapping the `b-jet Score` against the `light Score`. A `ReferenceLine` dynamically adjusts the b-jet score cutoff based on the `target_wp`, providing intuitive visual feedback for the mathematical Threshold.

---

## 4. Run Scripts & Training Pipeline (`scripts/run_regime.py`)

This file is the main driver running a single branch of the research grid.

*   **Line 25: `def main():`**
    Establishes CLI arguments (`--regime`, `--backbone`, `--wp`, `--seed`). This flexibility allows running sweeping ablation matrices via bash.
*   **Lines 40-56: Configuration Loading**
    It doesn't hardcode hyper-parameters. It loads YAML files merging `experiment.yaml`, `data.yaml`, the target `regime`, and the `backbone`.
*   **Line 66: Call to `train_pipeline`**
    Invokes the shared training logic from `src/piwpct/train.py` we analyzed earlier, yielding the fully trained PyTorch `model`.
*   **Lines 101-105: Exact Mondrian Conformal Calibration**
    THIS IS THE FINAL CORE INNOVATION. Once trained, the model is frozen. The script extracts logits directly from the isolated `CALIBRATION` split and passes them into the `MondrianConformalCalibrator`. This generates exact margin thresholds resulting in a statistically guaranteed $1 - \alpha$ coverage.
*   **Lines 107-119: Test Evaluation**
    Finally, the model passes entirely unseen true test data (`dataloaders["test"]`) into evaluation. Metrics such as `r_light` (light mistag), `eps_b` (b-efficiency), and Marginal Conformal Coverage are computed to validate the algorithm.

### Summary

PI-WPCT differentiates itself through:
1. **Mathematical Isolation**: `RiskAllocator.evaluate_validation_logits()` forces risk updates strictly on the validation split under `torch.no_grad()`. 
2. **Dynamic Adaptation**: `PIWPCTLoss` overrides `WPCTLoss` and simply accepts injected scalar parameters $\lambda$, protecting model gradients from downstream instability.
3. **Statistical Certainty**: `MondrianConformalCalibrator` runs on the separated `CALIBRATION` split ensuring exactly valid coverage on the test loop.
4. **MVP Integration**: The Next.js `page.tsx` and FastApi `main.py` is the prototype implementation allowing users to practically visualize the theoretical concepts via a physical target mapping to Recharts.
