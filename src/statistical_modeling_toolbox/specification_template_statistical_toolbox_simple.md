# Component: [Name] (Model/Likelihood/Prior)
> **Quick specification for statistical modeling components**  
> *Essential details to get started implementing statistical building blocks*

---

## 1. Statistical Definition
*What is this component? (1-2 sentences)*


*Mathematical form (use LaTeX):*
$$
% Main equation (e.g., probability density, model equation)
$$

*Parameters:*
- 

*Hyperparameters:*
- 

---

## 2. Properties
*Support (domain):*


*Key statistical properties:*
- Mean: 
- Variance: 
- Other: 

---

## 3. Use Cases
*When should this be used?*


*Typical applications:*
- 

---

## 4. Parametrization

### Standard Parametrization
*Conventional parameters (e.g., μ, σ² for Gaussian):*


### INLA Parametrization
*How are parameters expressed for DALIA/INLA?*


*Transformations needed:*
- 

---

## 5. Required Methods

### Core Methods (check what applies)
- [ ] `log_density(x, params)` - Log-density evaluation
- [ ] `gradient(x, params)` - Gradient w.r.t. parameters
- [ ] `hessian(x, params)` - Hessian matrix
- [ ] `precision_matrix(hyperparams)` - For models only
- [ ] `link_function()` - For likelihoods only
- [ ] `sample(n, params)` - Optional

*Key method signatures:*
```python
def log_density(self, x, params):
    """Compute log-density."""
    pass
```

---

## 6. Backend Requirements
*What matrix structures does this produce?*
- [ ] Dense
- [ ] Sparse (pattern: )
- [ ] Structured (type: )

*Linear algebra operations needed:*
- 

---

## 7. DALIA Integration
*Which DALIA modules use this component?*
- 

*What does DALIA need from this?*
- 

---

## 8. Validation
**Gradient check:**
- Method: Finite differences
- Tolerance: 

**Test case:**
- Known solution or comparison with: 

---

## 9. Usage Example
```python
# Basic usage example

```

---

## 10. Notes & References
*Literature references:*
- 

*Implementation notes:*
- 

*Open questions:*
- 
