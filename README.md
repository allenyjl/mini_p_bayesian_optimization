# 🧭 Flyer Propeller Optimization Project

## ⚡ Quick Reference

| Category | Variable / Concept | Description |
|-----------|-------------------|--------------|
| **Goal** | `Optimize propeller geometry` | Use Bayesian Optimization (BO) to find designs that maximize aerodynamic performance. |
| **Objective (to maximize)** | `ld_ratio` (lift / drag) | Derived from measured lift and drag. Alternative: `lift`, or `neg_drag` for minimizing drag. |
| **Design Variables (inputs)** | `camber`, `root_chord`, `trip_chord`, `corner_radius`, `angle_of_attack` | Controllable geometry parameters the optimizer can change. |
| **Side Metrics (measured outputs)** | `rpm`, `vibration` | Observed data from sensors — **not optimized** but correlated with performance. |
| **Model Type** | `MultiTaskGP` (Intrinsic Coregionalization Model) | Jointly models objective + side metrics to share information and improve predictions. |
| **Acquisition Function** | `qExpectedImprovement` | Optimized only on the **objective task (index 0)**. |
| **Bounds** | Min/max of design variables | Computed automatically from data. |
| **Output File** | `next_experiments_multitask.csv` | Suggested new propeller design parameters to test. |

---

## 🪶 Project Overview

This project develops a **Bayesian Optimization (BO)** workflow to improve the aerodynamic design of a **micro aerial vehicle propeller** (Mini Piccolissimo).  
The system learns from previous experimental results to propose new, improved design configurations for testing.

The workflow integrates:
- **Experimental data** (lift, drag, rpm, vibration)
- **Ax** and **BoTorch** for BO
- **Gaussian Process (GP)** surrogates for modeling the design–performance relationship

---

## 🧠 Core Objective

The main goal is to find optimal propeller geometries that maximize aerodynamic efficiency — typically measured by **Lift-to-Drag ratio (L/D)**.

### Inputs (Design Variables)
```
camber, root_chord, trip_chord, corner_radius, angle_of_attack
```

### Outputs
- Objective: `ld_ratio` (or `lift`, `neg_drag`)
- Side metrics: `rpm` (float) and `vibration` (0/1)

These side metrics are **measured** (not controlled), but contain useful correlations with performance.

---

## 🔍 Modeling Overview

### Single-Task GP (Original)
Used only one scalar function:
\[ y = f(x) + \epsilon \]
→ Could not incorporate `rpm` and `vibration` meaningfully.

### Multi-Task GP (Current)
Jointly models:
\[ f = \{ f_{ld}, f_{rpm}, f_{vibration} \} \]
with covariance:
\[ Cov(f_i(x), f_j(x')) = K_x(x, x') K_t(i, j) \]

- `K_x`: kernel over design space (geometry similarity)
- `K_t`: learned inter-task covariance (correlations among tasks)

Benefits:
- Shares information between tasks
- Handles missing data per task (partial observations)
- Provides better uncertainty estimates for acquisition

---

## 🧩 Technical Summary

| Component | Description |
|------------|--------------|
| **Frameworks** | Ax (experiment manager), BoTorch (BO core), GPyTorch (GP implementation), PyTorch (backend) |
| **Model** | `MultiTaskGP` with intrinsic coregionalization (ICM) |
| **Acquisition** | `qExpectedImprovement (qEI)` on the objective task only |
| **Data Inputs** | One row per experiment trial |
| **Outputs** | `next_experiments_multitask.csv` with new design candidates |
| **Noise** | Learned per-task noise parameters |
| **Standardization** | `Standardize(m=1)` per task to normalize outputs |
| **Partial Observations** | Automatically handled per task (drop NAs task-wise) |

---

## 🧮 Data Format Example

```csv
camber,root_chord,trip_chord,corner_radius,angle_of_attack,ld_ratio,rpm,vibration
0.05,1.2,1.0,0.08,5.2,10.5,3200,0
0.10,1.8,1.5,0.12,9.3,13.2,3550,1
...
```

- **Design variables:** controllable inputs (geometry)
- **ld_ratio:** main optimization target
- **rpm, vibration:** measured side metrics

---

## 🧠 How the BO Loop Works

1. **Data Input:** CSV with experimental results.  
2. **Model Fitting:** Fit a Multi-Task GP (`ld_ratio`, `rpm`, `vibration`).  
3. **Acquisition Optimization:** Compute `qEI` on the objective task only.  
4. **Suggestion:** Output new design points (geometry parameters only).  
5. **Experiment:** Fabricate/test suggested propellers.  
6. **Update:** Append new results and retrain.

---

## 🧩 Integration Notes

- Only provide bounds for **design variables** (min/max).  
- Do **not** provide bounds for `rpm` or `vibration` — they’re outputs.  
- `vibration` should be 0/1 (mapped from yes/no if needed).  
- If `rpm` or `vibration` columns are missing, switch to a `SingleTaskGP` (supported by the earlier notebooks).  
- You can extend to use `qNoisyExpectedImprovement` if measurements are very noisy.

---

## ⚙️ Example Usage (in Colab)

1. Open **`propeller_multitask_only.ipynb`** in Google Colab.
2. Run the setup cell to install libraries.
3. Upload your CSV file (must contain `ld_ratio`, `rpm`, `vibration`, and geometry columns).
4. The notebook will automatically detect:  
   - Objective column (`ld_ratio`)  
   - Design variables  
   - Side metrics  
5. Run all cells → download `next_experiments_multitask.csv`.

---

## ✅ Expected Outputs

- Trained **Multi-Task GP model**
- Optimized acquisition (`qEI`)
- `next_experiments_multitask.csv` with new proposed propeller designs

---

## 🧭 Purpose of This File

This `.md` file provides all necessary background for an **LLM** to understand the **Flyer Bayesian Optimization project**, its structure, and current implementation.

It should be uploaded or pasted into a new LLM session to reestablish project context without reloading previous conversations.

---
