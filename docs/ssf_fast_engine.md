# Fast frequentist SSF engine

`FrequentistSSF` defaults to a vectorized conditional-logit engine specialized to the canonical SSF design: every stratum contains exactly one chosen endpoint and a fixed number of alternatives.

For stratum `s` and candidate `j`, the conditional choice probability is

\[
P(Y_s=j)=\frac{\exp(x_{sj}^{T}\beta)}{\sum_k \exp(x_{sk}^{T}\beta)}.
\]

The corresponding log likelihood is

\[
\ell(\beta)=\sum_s\left[x_{s,y_s}^{T}\beta-\log\sum_j\exp(x_{sj}^{T}\beta)\right].
\]

This is exactly the same conditional-logit likelihood used previously; the new engine only evaluates it differently. The complete design is reshaped once to `(n_strata, n_choices, n_predictors)` and the likelihood, gradient, probabilities, and Hessian are evaluated with vectorized NumPy operations rather than Python-level iteration over strata.

The analytic score is

\[
\nabla\ell(\beta)=\sum_s\left[x_{s,y_s}-\sum_jp_{sj}x_{sj}\right],
\]

and the information matrix is

\[
I(\beta)=\sum_s\sum_j p_{sj}(x_{sj}-\mu_s)(x_{sj}-\mu_s)^T,
\qquad
\mu_s=\sum_jp_{sj}x_{sj}.
\]

The same information matrix supplies ordinary standard errors and the canonical CIIF diagnostic.

## Usage

The optimized engine is the default:

```python
fit = freq.fit()
```

or explicitly:

```python
fit = freq.fit(engine="fast")
```

The previous Statsmodels implementation remains available as a reference and compatibility engine:

```python
reference = freq.fit(engine="statsmodels")
```

For development or regression checking, coefficients and standard errors from the two engines should agree to numerical optimization tolerance.

Individual no-pooling fits use the same engine selection:

```python
individual = freq.fit_individuals(engine="fast")
```

The fast engine currently supports BFGS and L-BFGS-B (`method="bfgs"` or `method="lbfgs"`). Other Statsmodels optimizers remain available through `engine="statsmodels"`.

## Why this is faster

A general-purpose conditional-logit implementation must support groups with arbitrary numbers of successes and therefore performs more bookkeeping per group. Canonical SSF strata have a much simpler structure. Exploiting one chosen endpoint per stratum converts the likelihood directly into a dense categorical/softmax calculation, so tens of thousands of strata can be processed in compiled NumPy/SciPy operations.

The fast engine also vectorizes `choice_probabilities()` and `choice_scores()`. It uses `float64` for frequentist optimization and Hessian calculations, while Bayesian choice tensors keep their existing `float32` default.

## Validation

`tests/test_ssf_fast.py` fits the same synthetic choice data with both engines and compares coefficients, standard errors, and Hessians. This guards the important design principle that `engine="fast"` is an implementation optimization, not a different statistical estimator.
