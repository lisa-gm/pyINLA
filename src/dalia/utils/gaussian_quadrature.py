# Copyright 2024-2025 DALIA authors. All rights reserved.

from dalia import xp

from scipy.special import roots_hermite


# Dummy classes to avoid circular imports during testing
class DummyConfig:
    def __init__(self, alpha=2.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta

class DummyGammaPriorHyperparameters:
    def __init__(self, config):
        self.config = config
    
    def rescale_hyperparameters_to_internal(self, theta, direction):
        """Log transformation: external (positive) <-> internal (unconstrained)"""
        if direction == "forward":
            return xp.log(theta)
        elif direction == "backward":
            return xp.exp(theta)
        else:
            raise ValueError(f"Unknown direction: {direction}")

def compute_variance_gauss_hermite(mean_internal, variance_internal, transform, n_points=20):
    """
    Compute variance of transformed distribution using Gauss-Hermite quadrature
    
    For a distribution Y where log(Y) ~ N(μ, σ²), we want to compute:
    Var(Y) = E[Y²] - (E[Y])²
    
    Using Gauss-Hermite quadrature by transforming to standard normal form.
    
    Theory:
    If Z ~ N(0,1), then X = μ + σZ ~ N(μ, σ²)
    So Y = φ⁻¹(X) = φ⁻¹(μ + σZ) = exp(μ + σZ)
    
    E[Y] = E[exp(μ + σZ)] = exp(μ) E[exp(σZ)]
    E[Y²] = E[exp(2μ + 2σZ)] = exp(2μ) E[exp(2σZ)]
    
    Where E[exp(aZ)] can be computed using Gauss-Hermite quadrature.
    """
    
    # Get Gauss-Hermite quadrature points and weights
    nodes, weights = roots_hermite(n_points)
    
    # Transform nodes from Hermite polynomial roots to standard normal
    # Hermite nodes are for exp(-x²), we want exp(-x²/2)/√(2π)
    # So we scale by √2: z = √2 * node
    z_nodes = xp.sqrt(2) * nodes
    
    # Compute transformed values Y = φ⁻¹(μ + σZ) for each node
    internal_values = mean_internal + xp.sqrt(variance_internal) * z_nodes
    y_values = transform(internal_values, direction="backward")
    
    # Gauss-Hermite weights need to be adjusted for standard normal
    # Original: ∫ f(x) exp(-x²) dx ≈ Σ w_i f(x_i)
    # For standard normal: ∫ f(z) (1/√(2π)) exp(-z²/2) dz
    # After substitution x = z/√2: ∫ f(√2 x) (1/√π) exp(-x²) dx
    adjusted_weights = weights / xp.sqrt(xp.pi)
    
    # Compute first and second moments
    mean_y = xp.sum(adjusted_weights * y_values)
    second_moment_y = xp.sum(adjusted_weights * y_values**2)

    # Variance = E[Y²] - (E[Y])²
    variance_y = second_moment_y - mean_y**2
    
    return {
        'mean': mean_y,
        'second_moment': second_moment_y,
        'variance': variance_y,
        'std': xp.sqrt(variance_y)
    }


def test_gaussian_quadrature():
    """
    Comprehensive test suite for the Gaussian quadrature function.
    
    Tests multiple transformation functions and compares results with 
    analytical solutions where available.
    """
    import numpy as np
    
    print("=" * 80)
    print("COMPREHENSIVE GAUSSIAN QUADRATURE TESTS")
    print("=" * 80)
    
    # Test parameters
    test_tolerance = 1e-6
    n_quad_points = 50
    
    # Test 1: Identity transformation (should recover original normal moments)
    print("1. Testing Identity Transformation (X = Y)")
    print("-" * 50)
    
    def identity_transform(x, direction):
        return x  # No transformation
    
    mu = 2.0
    sigma2 = 1.5
    
    result = compute_variance_gauss_hermite(mu, sigma2, identity_transform, n_quad_points)
    
    # For identity, we should recover the original normal distribution moments
    expected_mean = mu
    expected_variance = sigma2
    
    print(f"  Expected mean: {expected_mean:.6f}, Got: {result['mean']:.6f}")
    print(f"  Expected var:  {expected_variance:.6f}, Got: {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - expected_mean)
    var_error = abs(result['variance'] - expected_variance)
    
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Var error:  {var_error:.2e}")
    
    assert mean_error < test_tolerance, f"Identity mean test failed: error = {mean_error}"
    assert var_error < test_tolerance, f"Identity variance test failed: error = {var_error}"
    print("  ✓ Identity transformation test PASSED")
    print()
    
    # Test 2: Log transformation (log-normal distribution)
    print("2. Testing Log Transformation (Y = exp(X))")
    print("-" * 50)
    
    def log_transform(x, direction):
        if direction == "forward":
            return xp.log(x)
        elif direction == "backward":
            return xp.exp(x)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    mu = 0.5
    sigma2 = 0.25
    
    result = compute_variance_gauss_hermite(mu, sigma2, log_transform, n_quad_points)
    
    # Analytical moments for log-normal: if log(Y) ~ N(μ, σ²)
    expected_mean = np.exp(mu + sigma2/2)
    expected_variance = (np.exp(sigma2) - 1) * np.exp(2*mu + sigma2)
    
    print(f"  Expected mean: {expected_mean:.6f}, Got: {result['mean']:.6f}")
    print(f"  Expected var:  {expected_variance:.6f}, Got: {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - expected_mean) / expected_mean
    var_error = abs(result['variance'] - expected_variance) / expected_variance
    
    print(f"  Relative mean error: {mean_error:.2e}")
    print(f"  Relative var error:  {var_error:.2e}")
    
    assert mean_error < 1e-4, f"Log-normal mean test failed: rel error = {mean_error}"
    assert var_error < 1e-4, f"Log-normal variance test failed: rel error = {var_error}"
    print("  ✓ Log transformation test PASSED")
    print()
    
    # Test 3: Linear transformation (Y = aX + b)
    print("3. Testing Linear Transformation (Y = aX + b)")
    print("-" * 50)
    
    a, b = 3.0, -1.5
    
    def linear_transform(x, direction):
        if direction == "forward":
            return (x - b) / a
        elif direction == "backward":
            return a * x + b
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    mu = 1.0
    sigma2 = 0.8
    
    result = compute_variance_gauss_hermite(mu, sigma2, linear_transform, n_quad_points)
    
    # For Y = aX + b where X ~ N(μ, σ²): E[Y] = aμ + b, Var[Y] = a²σ²
    expected_mean = a * mu + b
    expected_variance = a**2 * sigma2
    
    print(f"  Transform: Y = {a}X + {b}")
    print(f"  Expected mean: {expected_mean:.6f}, Got: {result['mean']:.6f}")
    print(f"  Expected var:  {expected_variance:.6f}, Got: {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - expected_mean)
    var_error = abs(result['variance'] - expected_variance)
    
    print(f"  Mean error: {mean_error:.2e}")
    print(f"  Var error:  {var_error:.2e}")
    
    assert mean_error < test_tolerance, f"Linear mean test failed: error = {mean_error}"
    assert var_error < test_tolerance, f"Linear variance test failed: error = {var_error}"
    print("  ✓ Linear transformation test PASSED")
    print()
    
    # Test 4: Quadratic transformation (Y = X²)
    print("4. Testing Quadratic Transformation (Y = X²)")
    print("-" * 50)
    
    def quadratic_transform(x, direction):
        if direction == "forward":
            return xp.sqrt(x)  # Only works for x >= 0
        elif direction == "backward":
            return x**2
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    mu = 0.0  # Centered to avoid issues with sqrt
    sigma2 = 0.5
    
    result = compute_variance_gauss_hermite(mu, sigma2, quadratic_transform, n_quad_points)
    
    # For Y = X² where X ~ N(0, σ²): E[Y] = σ², Var[Y] = 2σ⁴
    expected_mean = sigma2
    expected_variance = 2 * sigma2**2
    
    print(f"  X ~ N({mu}, {sigma2})")
    print(f"  Expected mean: {expected_mean:.6f}, Got: {result['mean']:.6f}")
    print(f"  Expected var:  {expected_variance:.6f}, Got: {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - expected_mean) / expected_mean
    var_error = abs(result['variance'] - expected_variance) / expected_variance
    
    print(f"  Relative mean error: {mean_error:.2e}")
    print(f"  Relative var error:  {var_error:.2e}")
    
    assert mean_error < 1e-3, f"Quadratic mean test failed: rel error = {mean_error}"
    assert var_error < 1e-2, f"Quadratic variance test failed: rel error = {var_error}"
    print("  ✓ Quadratic transformation test PASSED")
    print()
    
    # Test 5: Probit transformation (Y = Φ(X), where Φ is the standard normal CDF)
    print("5. Testing Probit Transformation (Y = Φ(X))")
    print("-" * 50)
    
    from scipy.stats import norm
    
    def probit_transform(x, direction):
        if direction == "forward":
            # Inverse probit: Φ⁻¹(x) = norm.ppf(x)
            # Clip to avoid numerical issues at boundaries
            x_clipped = xp.clip(x, 1e-15, 1-1e-15)
            return norm.ppf(x_clipped)
        elif direction == "backward":
            # Probit: Φ(x) = norm.cdf(x)
            return norm.cdf(x)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    # For probit transformation, we need to be careful about the domain
    mu_probit = 0.0  # Center at 0 for symmetry
    sigma2_probit = 0.5  # Moderate variance to avoid extreme values
    
    result = compute_variance_gauss_hermite(mu_probit, sigma2_probit, probit_transform, n_quad_points)
    
    # For Y = Φ(X) where X ~ N(0, σ²), we can compute this numerically
    # Since there's no closed form, we'll use a high-precision reference calculation
    
    # Reference calculation using many quadrature points
    ref_result = compute_variance_gauss_hermite(mu_probit, sigma2_probit, probit_transform, 200)
    
    print(f"  X ~ N({mu_probit}, {sigma2_probit})")
    print(f"  Reference mean (200 pts): {ref_result['mean']:.6f}")
    print(f"  Test mean ({n_quad_points} pts):     {result['mean']:.6f}")
    print(f"  Reference var (200 pts):  {ref_result['variance']:.6f}")
    print(f"  Test var ({n_quad_points} pts):      {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - ref_result['mean']) / ref_result['mean']
    var_error = abs(result['variance'] - ref_result['variance']) / ref_result['variance']
    
    print(f"  Relative mean error: {mean_error:.2e}")
    print(f"  Relative var error:  {var_error:.2e}")
    
    # For probit, we expect the mean to be close to 0.5 due to symmetry
    expected_mean_approx = 0.5
    mean_deviation = abs(result['mean'] - expected_mean_approx)
    
    print(f"  Expected mean ≈ 0.5, deviation: {mean_deviation:.4f}")
    
    assert mean_error < 1e-3, f"Probit mean test failed: rel error = {mean_error}"
    assert var_error < 1e-2, f"Probit variance test failed: rel error = {var_error}"
    assert mean_deviation < 0.1, f"Probit mean should be close to 0.5: deviation = {mean_deviation}"
    print("  ✓ Probit transformation test PASSED")
    print()
    
    # Test 6: Gamma prior rescaling
    print("6. Testing Gamma Prior Rescaling")
    print("-" * 50)
    
    config = DummyConfig(alpha=2.0, beta=1.0)
    gamma_prior = DummyGammaPriorHyperparameters(config=config)
    
    def gamma_rescale(x, direction):
        return gamma_prior.rescale_hyperparameters_to_internal(x, direction)
    
    mu = 0.2
    sigma2 = 0.3
    
    result = compute_variance_gauss_hermite(mu, sigma2, gamma_rescale, n_quad_points)
    
    # This is the same as log-normal since rescale uses exp transformation
    expected_mean = np.exp(mu + sigma2/2)
    expected_variance = (np.exp(sigma2) - 1) * np.exp(2*mu + sigma2)
    
    print(f"  Expected mean: {expected_mean:.6f}, Got: {result['mean']:.6f}")
    print(f"  Expected var:  {expected_variance:.6f}, Got: {result['variance']:.6f}")
    
    mean_error = abs(result['mean'] - expected_mean) / expected_mean
    var_error = abs(result['variance'] - expected_variance) / expected_variance
    
    print(f"  Relative mean error: {mean_error:.2e}")
    print(f"  Relative var error:  {var_error:.2e}")
    
    assert mean_error < 1e-4, f"Gamma rescale mean test failed: rel error = {mean_error}"
    assert var_error < 1e-4, f"Gamma rescale variance test failed: rel error = {var_error}"
    print("  ✓ Gamma prior rescaling test PASSED")
    print()
    
    # Test 7: Convergence with increasing quadrature points
    print("7. Testing Convergence with Quadrature Points")
    print("-" * 50)
    
    mu_conv = 0.1
    sigma2_conv = 0.4
    expected_mean_conv = np.exp(mu_conv + sigma2_conv/2)
    
    n_points_list = [5, 10, 15, 20, 30, 50, 75]
    errors = []
    
    print("  n_points  |   Mean      |  Rel. Error")
    print("  ----------|-------------|------------")
    
    for n in n_points_list:
        result = compute_variance_gauss_hermite(mu_conv, sigma2_conv, log_transform, n)
        rel_error = abs(result['mean'] - expected_mean_conv) / expected_mean_conv
        errors.append(rel_error)
        
        print(f"  {n:8d}  | {result['mean']:10.6f} | {rel_error:.3e}")
    
    # Check that errors generally decrease (allowing some numerical noise)
    improving = sum(errors[i+1] < errors[i] * 1.1 for i in range(len(errors)-1))
    improvement_rate = improving / (len(errors) - 1)
    
    print(f"  Improvement rate: {improvement_rate:.1%}")
    assert improvement_rate > 0.6, f"Convergence test failed: improvement rate = {improvement_rate}"
    print("  ✓ Convergence test PASSED")
    print()
    
    # Test 8: Edge cases
    print("8. Testing Edge Cases")
    print("-" * 50)
    
    # Small variance
    result_small = compute_variance_gauss_hermite(1.0, 1e-6, log_transform, 20)
    expected_small = np.exp(1.0)  # When σ² → 0, E[exp(X)] → exp(μ)
    error_small = abs(result_small['mean'] - expected_small) / expected_small
    
    print(f"  Small variance test: rel error = {error_small:.3e}")
    assert error_small < 1e-3, f"Small variance test failed: rel error = {error_small}"
    
    # Large variance (but not too large to avoid overflow)
    result_large = compute_variance_gauss_hermite(0.0, 2.0, log_transform, 100)
    expected_large = np.exp(1.0)  # E[exp(X)] = exp(μ + σ²/2) = exp(0 + 2/2) = e
    error_large = abs(result_large['mean'] - expected_large) / expected_large
    
    print(f"  Large variance test: rel error = {error_large:.3e}")
    assert error_large < 1e-2, f"Large variance test failed: rel error = {error_large}"
    
    print("  ✓ Edge cases test PASSED")
    print()
    
    # Summary
    print("=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)


if __name__ == "__main__":
    test_gaussian_quadrature()