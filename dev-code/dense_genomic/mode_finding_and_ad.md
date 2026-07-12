## How Mode-Finding Breaks AD

This is an important architectural constraint. In Gaussian likelihoods, the INLA objective is a closed-form computation — everything is differentiable. But for non-Gaussian likelihoods (Poisson, binomial, etc.), the latent field's mode must be found numerically via Newton-Raphson or similar:

```python
def find_mode(y, theta, Q_prior):
    """Newton-Raphson to find the mode of the posterior."""
    x = x0
    for _ in range(max_iter):
        gradient = compute_gradient(x, y, theta, Q_prior)
        hessian = compute_hessian(x, y, theta, Q_prior)
        x = x - solve(hessian, gradient)  # x = x - H⁻¹g
    return x
```

**Why this breaks AD:**

1. **The mode-finding loop is an iterative algorithm, not a closed-form expression.** AD systems (like Autograd, JAX, PyTorch) trace operations through a computation graph. An iterative loop with a convergence check (`while ||gradient|| > tol`) creates a dynamic computation graph that depends on the data, not just the inputs.

2. **The number of iterations is data-dependent.** AD systems need a fixed computation graph. If Newton takes 5 iterations for one HP point and 50 for another, the traced graph changes shape — most AD systems can't handle this.

3. **The convergence check is non-differentiable.** The `while ||g|| > tol` condition is a discontinuous function of the inputs. The gradient is undefined at the boundary.

4. **Solving the linear system inside Newton** (the `H⁻¹g` step) creates a nested optimization. AD through a linear solver is possible in theory but extremely expensive — you're differentiating through an entire solve operation.

5. **The mode `x*` is an implicit function of the HPs** (defined by the fixed-point equation `∇log p(x*, θ) = 0`). Differentiating through it requires the implicit function theorem: `∂x*/∂θ = -H⁻¹ ∂²log p/∂x∂θ`. This is exactly what INLA does analytically — but AD doesn't know this shortcut.

**Practical consequence:** For non-Gaussian likelihoods, AD through the full INLA objective is either impractical or requires significant workarounds (e.g., unrolling a fixed number of Newton iterations, using implicit differentiation). Finite differences is often the more pragmatic choice for the full INLA objective, even if individual components are differentiable.

**However:** If you only need the gradient of the *Gaussian* part (or a Laplace-approximated part that is closed-form), AD works fine there. The mode-finding breakage only affects the non-Gaussian likelihood terms.
