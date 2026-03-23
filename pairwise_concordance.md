# Pairwise concordance loss

## Problem

Given two vectors per sample, $a, b \in \mathbb{R}^D$, measure whether they agree on the ordering of elements. If $a_i > a_j$, does $b_i > b_j$?

This is a differentiable relaxation of Kendall's tau, designed for gradient-based optimisation with decoupled per-pair gradients.


## Loss definition

For each pair $(i, j)$ with $i < j$, define the concordance product:

$$p_{ij} = s \cdot \delta a_{ij} \cdot \delta b_{ij}$$

where $s$ is the `scale` parameter and $\delta a_{ij}, \delta b_{ij}$ are pairwise differences (see below). Concordant pairs have $p_{ij} > 0$; discordant pairs have $p_{ij} < 0$.

**Hinge mode:**

$$\ell_{ij} = \max(0,\; 1 - p_{ij})$$

Zero when $p_{ij} \geq 1$. Piecewise linear, sparse gradients. Preferred for multi-task uncertainty weighting because the loss can reach exact zero.

**Logistic mode:**

$$\ell_{ij} = \log(1 + e^{-p_{ij}})$$

Smooth everywhere, nonzero floor. Gradient never vanishes for discordant pairs.

**Per-sample loss** (exact path):

$$L_n = \frac{1}{\binom{D}{2}} \sum_{i < j} \ell_{ij}$$


## Difference modes

### Absolute differences

$$\delta a_{ij} = a_i - a_j, \qquad \delta b_{ij} = b_i - b_j$$

Scale depends on input magnitude. Simple, no hyperparameters beyond `scale`.

### Relative differences (detached denominator)

$$\delta a_{ij} = \frac{2(a_i - a_j)}{(|a_i| + |a_j| + c)^{\mathrm{stop}}}$$

where $(\cdot)^{\mathrm{stop}}$ denotes stop-gradient (detached from the computation graph).

**Forward pass:** exact relative difference, bounded in $[-2, 2]$, producing $O(1)$ products regardless of input scale.

**Backward pass:** the gradient with respect to $a_i$ is:

$$\frac{\partial \delta a_{ij}}{\partial a_i} = \frac{2}{d_{ij}^{\mathrm{stop}}}$$

where $d_{ij} = |a_i| + |a_j| + c$. This is a positive constant per pair — no cross-element coupling, no sign dependence, bounded at $2/c$.


## Why detach the denominator

Without detach, the full quotient-rule gradient is:

$$\frac{\partial}{\partial a_i}\frac{2(a_i - a_j)}{d_{ij}} = \frac{2d_{ij} - 2(a_i - a_j)\,\mathrm{sign}(a_i)}{d_{ij}^2}$$

This has three problems:

### 1. Gradient suppression of large-difference pairs

For a discordant pair where $a_i \gg a_j > 0$: $\mathrm{numer} = 2(a_i + a_j) - 2(a_i - a_j) = 4a_j$, $\mathrm{denom} = (a_i + a_j)^2$. The gradient is $\approx 4a_j / a_i^2 \to 0$ as $a_i$ grows. The quotient rule suppresses gradients for the pairs with the largest discordance — exactly the ones that most need correction.

With detach: $2 / (a_i + a_j)^{\mathrm{stop}}$. Still $O(1/a_i)$ but not $O(a_j/a_i^2)$. The upstream concordance loss gradient carries the full correction signal without the denominator cancelling it.

### 2. Cross-element coupling

Without detach, $\partial \delta a_{ij} / \partial a_i$ depends on $a_j$ through the denominator. This creates off-diagonal Hessian structure: perturbing $a_j$ changes the gradient for $a_i$ even though they're different elements. With detach, the denominator is a frozen constant per pair, so $a_j$'s value in the denominator doesn't affect $a_i$'s gradient.

### 3. Near-zero divergence

When $a_i \approx a_j \approx 0$, both paths diverge as $O(1/\epsilon)$:

- Without detach: $(2 \cdot 2\epsilon - 0) / (2\epsilon)^2 = 1/\epsilon$
- With detach: $2 / 2\epsilon = 1/\epsilon$

The parameter $c$ floors the denominator at $c$, bounding the gradient at $2/c$ in both cases. This is needed regardless of detach.

### Geometric interpretation

The detach replaces the curved Riemannian metric induced by the normalisation with a flat metric. Without detach, the implicit metric assigns high cost to moving parameters in "saturated" directions (where the relative difference is already large). With detach, all directions have equal cost — the normalisation is a forward-pass feature for correct scaling, not a backward-pass constraint on the optimisation geometry.

This is equivalent to preconditioned gradient descent with preconditioner:

$$P = \mathrm{diag}\left(\frac{d_{ij}}{d_{ij} - (a_i - a_j)\,\mathrm{sign}(a_i)}\right)$$

which removes the quotient-rule suppression while preserving the descent direction.


## The `c` parameter

Minimum meaningful pair magnitude. Sets the gradient ceiling at $2/c$.

For gene expression / metabolic flux data where values are $O(1)$ after normalisation, $c = 0.1$ means:
- Pairs with $|a_i| + |a_j| \geq 0.1$: gradient $\leq 20$
- Pairs with $|a_i| + |a_j| < 0.1$: gradient clamped, treated as noise

The `c` parameter is independent of `delta` (binning resolution) and `scale` (loss sensitivity). It controls only the backward-pass scaling for the relative difference mode.


## Scale parameter

The `scale` parameter $s$ multiplies the concordance product before the activation:

$$p_{ij} = s \cdot \delta a_{ij} \cdot \delta b_{ij}$$

| Mode | Activation | Scale controls |
|------|-----------|----------------|
| hinge | $\mathrm{relu}(1 - s \cdot \delta a \cdot \delta b)$ | threshold: loss = 0 when $\delta a \cdot \delta b \geq 1/s$ |
| logistic | $\mathrm{softplus}(-s \cdot \delta a \cdot \delta b)$ | steepness: larger $s$ concentrates gradient on near-threshold pairs |

This unifies the previously separate `alpha` (product multiplier) and `margin` (hinge threshold) parameters. The equivalence: $\mathrm{relu}(m - \alpha x) = m \cdot \mathrm{relu}(1 - ({\alpha}/{m}) x)$, so $s = \alpha/m$ with a constant factor $m$ absorbed into the task weight.


## Gradient structure

### Per-element gradient (exact path, hinge mode, absolute diff)

$$\frac{\partial L_n}{\partial b_k} = \frac{1}{\binom{D}{2}} \sum_{j \neq k} \frac{\partial \ell_{kj}}{\partial b_k}$$

For a single pair $(k, j)$:

$$\frac{\partial \ell_{kj}}{\partial b_k} = \begin{cases} -s \cdot \delta a_{kj} & \text{if } 1 - s \cdot \delta a_{kj} \cdot \delta b_{kj} > 0 \\ 0 & \text{otherwise} \end{cases}$$

Each pair contributes independently. The gradient for $b_k$ depends only on pairs involving $k$, not on other elements' values. This is the key structural difference from Spearman/Pearson correlation losses, where the standardisation Jacobian $\partial z_i / \partial r_j = (1/\sigma)(\delta_{ij} - 1/M - z_i z_j / M)$ couples all elements.

### Hessian off-diagonal

The off-diagonal Hessian $H_{kl} = \partial^2 L / \partial b_k \partial b_l$ for $k \neq l$:

**Hinge:** $H_{kl} = 0$ everywhere except at the relu kink where $p_{kl} = 1$ exactly. Piecewise linear → piecewise constant gradient → zero second derivative almost everywhere. After optimisation, some pairs land near the kink; finite-difference probes across the kink pick up small nonzero $H_{kl}$. This is an artefact of non-differentiability at a measure-zero set, not true coupling.

**Logistic:** $H_{kl} = -s^2 \cdot (\delta a_{kl})^2 \cdot \sigma'(-p_{kl})$ where $\sigma'$ is the logistic sigmoid derivative. Small but nonzero — smooth curvature from the sigmoid.

Empirically (D=64, random data): hinge $|H_{\mathrm{od}}| \approx 0$, logistic $|H_{\mathrm{od}}| \approx 10^{-4}$. Compare Spearman: $|H_{\mathrm{od}}| \approx 10^{-3}$ (constant, never diminishes).


## Computation strategies

### Exact: $O(D^2)$

Materialise all $\binom{D}{2}$ pairs. Fine for $D \lesssim 1000$.

### Binned: $O(K^2)$

1. Quantise at resolution $\delta$: $\hat{a}_i = \mathrm{round}(a_i / \delta)$, same for $b$.
2. Identify unique $(\hat{a}, \hat{b})$ cells. Let $K$ = number of occupied cells.
3. Compute cell means $\bar{a}_r, \bar{b}_r$ via scatter-add (differentiable; cell assignment treated as constant).
4. Pairwise loss between $K$ cell means, weighted by count products $n_r \cdot n_s$.
5. Pairs within the same cell are tied (zero difference) → skipped by upper-triangle mask.

Gradient flows through cell means to individual elements: $\partial \bar{a}_r / \partial a_i = 1/n_r$ if element $i$ belongs to cell $r$, else 0.

Cell assignment is locally stable: small perturbations do not change which cell an element belongs to (unless it crosses a bin boundary). At boundaries, the assignment is piecewise constant and the gradient is defined by the current cell.

Note: the binned path is currently batch-serial (Python loop over samples). For large batch sizes this is the main performance bottleneck; a fused implementation would need batch-aware unique/scatter ops.

### Binned + sampled: $O(n_{\mathrm{pairs}})$

Instead of all $\binom{K}{2}$ cell pairs, sample $n_{\mathrm{pairs}}$ pairs via factored categorical sampling:

1. Draw $r \sim \mathrm{Categorical}(n_1/N, \ldots, n_K/N)$
2. Draw $s \sim \mathrm{Categorical}(n_1/N, \ldots, n_K/N)$ independently
3. Reject if $r \geq s$. Accept rate $\approx 50\%$; oversample $3\times$.

This samples pair $(r, s)$ with probability $\propto n_r \cdot n_s$, matching the exhaustive weighting. Conditioned on drawing at least one valid upper-triangle pair, the estimator is unbiased:

$$\mathbb{E}[\hat{L}] = \frac{\sum_{r<s} w_{rs} \ell_{rs}}{\sum_{r<s} w_{rs}} = L_{\mathrm{exhaustive}}$$

In the current implementation, rejection sampling is oversampled $3\times$ and returns a graph-connected zero if no valid pair is drawn. That fallback is mainly defensive and should be rare when $K$ is not tiny, but it does introduce a small downward bias in those rare cases.

Variance decreases as $O(1/n_{\mathrm{pairs}})$. Away from the empty-sample fallback, the gradient estimator is likewise unbiased by linearity of expectation.

Rule of thumb: $n_{\mathrm{pairs}} \approx K$ gives decent convergence; below $K/2$ noise dominates.


## Relationship to Kendall's tau

Kendall's tau:

$$\tau = \frac{C - D}{\binom{D}{2}}$$

where $C$ = concordant pairs, $D$ = discordant. The hinge loss with large `scale` approximates $(1 - \tau)/2$: discordant pairs contribute loss $\approx 1$, concordant pairs contribute $\approx 0$.

**Daniels' inequality** relates tau to Spearman's rho:

$$\frac{3\tau - 1}{2} \leq \rho_S \leq \frac{3\tau + 1}{2}$$

Empirically: $\rho_S \approx 1.5\tau - 0.5\tau^3$. Optimising $\tau \geq 0.8$ guarantees $\rho_S \geq 0.70$.

**Hinge loss as a progress metric.** The hinge loss value does not track rho linearly. A model can have $\rho = 0.5$ (decent correlation) while most pairs are within the margin band, keeping the loss high. Use rho as the monitoring metric; the loss is a training signal with correct gradients, not a progress indicator.


## Degenerate cases

All degenerate paths return graph-connected zeros:

```python
a.sum(dim=-1) * 0 + b.sum(dim=-1) * 0
```

Mathematically zero with zero gradient, but maintains autograd graph connection to both inputs. Downstream code (uncertainty weighting, gradient logging) never sees a detached scalar.

Degenerate conditions:
- $D < 2$: no pairs exist.
- $K < 2$ (binned): all elements in one cell (fully tied).
- Total pair weight $< 1$ (binned exhaustive): all weight on the diagonal.
- No valid samples drawn (binned sampled): rejection sampling yielded nothing.


## API

```python
pairwise_concordance(
    a, b,                    # (batch, D) tensors
    mode="hinge",            # "hinge" or "logistic"
    scale=1.0,               # product multiplier (both modes)
    diff="absolute",         # "absolute" or "relative"
    delta=None,              # None=exact, >0=binned (must be strictly positive if set)
    n_pairs=None,            # None=exhaustive, >0=sampled (requires delta)
    c=0.1,                   # gradient floor (only for diff="relative")
) -> (batch,)               # per-sample loss
```

Three orthogonal parameter groups:
- **Loss shape:** `mode` + `scale`
- **Difference scaling:** `diff` + `c` (c only relevant for relative)
- **Computation budget:** `delta` + `n_pairs` (n_pairs only relevant with delta)

For typical usage: `pairwise_concordance(a, b)` uses hinge mode with absolute differences and exact computation. The only parameter that affects training dynamics is `scale`.

