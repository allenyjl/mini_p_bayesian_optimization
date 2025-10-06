
# Flyer ML Optimization — Bayesian Optimization with Multi‑Task GP (Ax/BoTorch)

This README explains the end‑to‑end design of the rebuilt Bayesian Optimization (BO) pipeline for the **Flyer** project, with an emphasis on **why** each choice was made and **how** to adapt it. The implementation uses **BoTorch**/**GPyTorch** directly (easily wrapped in Ax later) and treats **RPM** and **vibration** as **measured side metrics** rather than optimizable inputs. The surrogate is a **Multi‑Task Gaussian Process (MTGP)** so information transfers between the main objective and side metrics, even with **partial observations**.

> **TL;DR**: We only optimize over **design parameters** (e.g., geometry), while we **jointly model** the objective + side metrics (RPM, vibration). The acquisition function is optimized **only over the objective task**. You **do not** provide bounds for RPM/vibration—only for design variables.

---

## Table of Contents

1. Problem Framing  
2. Data Expectations  
3. Modeling: Why Multi‑Task GP?  
4. Training Tensors & Task Feature  
5. Fitting the MTGP  
6. Acquisition: qEI on Objective Task  
7. Bounds & Variables You Must Set  
8. Partial Observations  
9. Vibration as Binary  
10. Noise & Standardization  
11. Batching, qNEI, Constraints  
12. Evaluation & Diagnostics  
13. Colab Setup & Running  
14. Integrating with Ax  
15. Extending or Modifying the Approach  
16. Common Pitfalls & Troubleshooting  
17. FAQ  

---

## 1) Problem Framing

- **Goal:** Suggest new **design parameter** settings (e.g., prop/wing geometry) that improve a target **objective** (e.g., `ld_ratio`, `lift`, or `efficiency`).  
- **Side metrics:** The dataset also contains **RPM** and **vibration**. These are **measured** during experiments, not under our control, and **should not** be optimized. We want to use them to improve the fidelity of objective predictions.
- **Key modeling choice:** Use a **Multi‑Task GP** where the tasks are:
  - `objective` (e.g., `ld_ratio`) — *the only task we optimize the acquisition for*
  - `rpm` (optional)
  - `vibration` (optional)

This transforms the regression into **multi‑output** modeling with **shared structure** between tasks. That lets the model leverage correlations like “lift increases with RPM” or “high vibration inflates measurement noise,” even if some rows are missing a side metric.

---

## 2) Data Expectations

Your dataframe **should** contain:

- **Design variables** (continuous/categorical encoded): `DESIGN_VARS = [...]`  
- **Objective column**: `OBJECTIVE_COL` (scalar numeric)  
- **Side metrics** (0..N of them): `SIDE_TASKS = ['rpm', 'vibration', ...]`

**Notes:**
- RPM should be numeric.  
- Vibration should be mapped to **0/1** (if strings like “yes/no”, map to 1/0).  
- Missing values are allowed (we will **drop NAs per task** during tensor build).

---

## 3) Modeling: Why Multi‑Task GP?

- A **single‑task GP** assumes one scalar function \( y = f(x) \). Adding RPM/vibration as columns in `x` would incorrectly treat them as **controllable inputs**.

- A **multi‑task GP** models a **vector‑valued function** \(\mathbf{f}(x) = [f_{\text{objective}}(x), f_{\text{rpm}}(x), f_{\text{vibration}}(x)]\) with **inter‑task correlations**.  

- This is implemented via the **Intrinsic Coregionalization Model (ICM)** in `MultiTaskGP`, with covariance:

  \[ \mathrm{Cov}(f_i(x), f_j(x')) = K_x(x, x') \cdot K_t(i, j) \]

  where \(K_x\) is the input kernel and \(K_t\) is the **learned task covariance**.


### Benefits
- **Partial observability:** Rows can be missing RPM/vibration; the MTGP still learns from rows where they exist.

- **Information sharing:** Correlated tasks reduce uncertainty for the objective.

- **No bounds for side metrics:** We don’t optimize over them; they are outputs.


---

## 4) Training Tensors & Task Feature

We build training data by **stacking** rows **per task** and appending a **task index** as the last column of `X`:

- Task indices:  
  - `0` → `OBJECTIVE_COL`  
  - `1` → `'rpm'` (if present)  
  - `2` → `'vibration'` (if present)  

**Process:**
1. For each task `t`:

   - Take `df[DESIGN_VARS + [t]]`, drop NA in `t`.

   - `X = design_values`; `Y = t_values`.

   - Append a column of the constant **task index** for task `t` to `X`.

2. Concatenate across tasks → `(X_all, Y_all)` for the MTGP.


This is exactly what `MultiTaskGP` expects when using a **task feature**.

---

## 5) Fitting the MTGP

- **Model:** `MultiTaskGP(train_X=X_all, train_Y=Y_all, task_feature=len(DESIGN_VARS), outcome_transform=Standardize(m=1))`  
- **Loss:** `ExactMarginalLogLikelihood`  
- **Fitting:** `fit_gpytorch_model(mll)`

**Standardize outcomes?** Usually helpful; set `STANDARDIZE_Y=True` (default in the notebook) to stabilize training across tasks with different scales.

---

## 6) Acquisition: qEI on Objective Task

We **optimize the acquisition only for the objective**, not for side metrics. To do this cleanly:

- Wrap the MTGP so **any query `X`** (design vars only) gets an **objective task index (0)** appended internally.  
- Use **qExpectedImprovement** with `best_f` computed from **objective task observations only**.  
- Optimize over the **design‑space bounds** (see next section).

This yields a set of **candidate design points** — the next experiments — **without** asking the model to predict or set RPM/vibration.

**Why qEI?** It’s a standard, robust choice for noiseless or mildly noisy settings. If your data are quite noisy or you’re proposing **batches**, use **qNEI**.

---

## 7) Bounds & Variables You Must Set

You **must** specify bounds **only** for **design variables**:

```python
DESIGN_VARS = ["pitch", "chord", "radius"]
BOUNDS = {
  "pitch": (10.0, 40.0),
  "chord": (1.0, 3.0),
  "radius": (5.0, 20.0)
}
```

- **No bounds** for RPM/vibration — not inputs.
- These bounds define the **search region** during acquisition optimization.

---

## 8) Partial Observations

**Allowed and expected.** For each task, we drop NA **only for that task’s column**. The resulting tensors can have **different row counts per task**. The MTGP’s learned **task covariance** makes use of all available signals to improve predictions for the **objective**.

Practical tip: If a side metric is missing for **most** rows, it may contribute little. You can drop it from `SIDE_TASKS` to simplify.

---

## 9) Vibration as Binary

- We treat `vibration` as **numeric 0/1** in regression. This is a pragmatic choice to keep a **single MTGP**.
- If you truly need probabilistic classification (e.g., Bernoulli likelihood), you can build a composite model (separate GP heads) and combine with `ModelListGP`. However, you’ll lose ICM correlations between regression head(s) and classification head out‑of‑the‑box.

**Recommendation:** Start with 0/1 regression. If needed later, experiment with a composite approach.

---

## 10) Noise & Standardization

- **Noise:** The MTGP learns observation noise. If you have per‑point noise estimates, you can pass them (advanced).  
- **Standardization:** `Standardize(m=1)` per task often stabilizes training when tasks have different output scales (RPM vs L/D vs binary).

---

## 11) Batching, qNEI, Constraints

- **Batch suggestions:** Set `q>1` in `optimize_acqf` to propose multiple designs **jointly** (correlated batch).  
- **qNEI:** If observations are noisy or you want to be robust to noise, switch to `qNoisyExpectedImprovement`.  
- **Constraints:** If your design space has feasibility rules (e.g., geometry consistency), pass `inequality_constraints` to `optimize_acqf` or filter candidates post hoc. For complex feasibility, consider a feasibility classifier.

---

## 12) Evaluation & Diagnostics

- **Compare proposals** from MTGP vs a **single‑task GP** baseline (objective‑only). The notebook writes an optional baseline CSV.

- **Posterior checks:** Inspect posterior means/variances for objective at known points; are they sensible?

- **Sanity tests:** With synthetic data, see if MTGP recovers the optimum faster than single‑task when side signals are informative.


**Offline metrics (if you have a holdout set):**
- RMSE / MAE on the objective
- Negative log likelihood
- Calibration checks with posterior variances

---

## 13) Colab Setup & Running

1. Upload `flyer_multitask_bo.ipynb` to Google Colab.  
2. If on a fresh runtime, uncomment the `pip install` line in **cell 0**.  
3. Configure in **cell 1**:

   - `DATA_CSV` → path to your CSV (or keep `None` and use the synthetic demo)

   - `DESIGN_VARS` and `BOUNDS`

   - `OBJECTIVE_COL`, choose side tasks present in your data

4. Run top‑to‑bottom.  
5. You get `next_experiments_multitask_gp.csv` with proposed designs (and optional single‑task baseline CSV).


---

## 14) Integrating with Ax

Wrap the fitted BoTorch model in an Ax `BotorchModel` by:
- Providing a custom **model bridge** that ensures the **task feature is fixed to 0** (objective) for acquisition.  
- Mapping Ax parameter space → design tensors, and vice versa.  
- Implement `predict` to append the objective task index before querying the posterior.

Because Ax’s standard `SingleTaskGP` doesn’t model multi‑output with ICM, use a **custom BoTorch model wrapper** in Ax for multi‑task behavior.

---

## 15) Extending or Modifying the Approach

- **Contextual single‑task (fallback):** If you can’t do multi‑task, you *could* treat RPM/vibration as *context variables* (inputs) **only if** you have them at suggestion time. Not your case; prefer MTGP.

- **Heteroskedastic noise:** If measurement noise varies with vibration, consider heteroskedastic models or noise conditioned on features (advanced).

- **Multi‑fidelity:** If you add simulated metrics, switch to a multi‑fidelity GP (fidelity kernels or tasks).  

- **Custom kernels:** If geometry induces periodicity/symmetry, experiment with periodic kernels or ARD per dimension.


---

## 16) Common Pitfalls & Troubleshooting

- **Forgetting bounds:** The optimizer needs **bounds for every design var** — and **only** for design vars.  
- **NA explosions:** Drop NA **per task**, not globally.  
- **Vibration labels:** Map to 0/1 before training.  
- **Acquisition task:** Ensure the wrapper **appends task=0** so you optimize for the objective.  
- **Scaling:** Keep `STANDARDIZE_Y=True`; consider log transforms for skewed outputs.


---

## 17) FAQ

**Q1. Do we need bounds for RPM or vibration?**  

**A. No.** They are **outputs**, not inputs. Provide bounds only for **design parameters**.


**Q2. How does the model use RPM/vibration if we don’t optimize over them?**  

The MTGP learns **correlations** between the objective and the side tasks via a **task covariance** matrix. Observing RPM/vibration helps reduce uncertainty in the objective function, improving predictions and acquisition.


**Q3. What if some rows don’t have RPM or vibration?**  

That’s fine. We build training tensors **per task** by dropping NA **only for that task**. The MTGP still shares information across tasks that *do* have data.


**Q4. Why not put RPM/vibration into the input `x`?**  

Because they are **not controllable** and are **observed after** the design is chosen. Treating them as inputs would imply we need them at suggestion time (we don’t) and misrepresents the data‑generating process.


**Q5. When should I use qNEI?**  

Use **qNEI** when your measurements are **noisy** or when you propose **batches** and need robustness to observation noise. It generally performs better than qEI in noisy settings.


---

### Outputs
- `next_experiments_multitask_gp.csv` — proposed designs from MTGP + qEI (objective only).  
- `next_experiments_single_task_gp.csv` — (optional) single‑task baseline proposals.

Append new trial results to your CSV and re‑run to iterate.
