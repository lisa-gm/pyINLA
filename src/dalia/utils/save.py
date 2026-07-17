"""Utilities for saving DALIA results."""

import json
import os
from typing import Dict, List, Optional, Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy arrays and other numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.generic)):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        return super().default(obj)


def save_to_json(
    results: Dict[str, Any],
    filename: str,
    hyperparameter_names: Optional[List[str]] = None,
    items_to_save: Optional[List[str]] = None,
) -> None:
    """
    Save DALIA inference results to a JSON file.

    This function saves DALIA results in a JSON-compatible format, automatically
    converting numpy arrays to lists for JSON serialization.

    Parameters
    ----------
    results : dict
        Dictionary containing DALIA inference results. Expected keys include:
        - 'theta': external hyperparameters
        - 'theta_internal': internal (transformed) hyperparameters
        - 'x': latent parameters
        - 'cov_theta_internal': covariance matrix of hyperparameters
        - 'marginals_hp': marginal distributions of hyperparameters (optional)
    
    filename : str
        Path to output JSON file. If directory doesn't exist, it will be created.
    
    hyperparameter_names : list of str, optional
        Names of hyperparameters to extract from results['marginals_hp'].
        If None, all hyperparameters in marginals_hp will be saved.
        Example: ['tau', 'phi', 'prec_o']
    
    items_to_save : list of str, optional
        Which result items to save. If None, all items are saved.
        Possible values: 'theta_internal', 'theta_external', 'x',
                        'cov_theta_internal_diagonal', 'cov_theta_internal_full',
                        'hyperparameters'
        Example: ['theta_external', 'x', 'hyperparameters']

    Returns
    -------
    None

    Examples
    --------
    >>> # Save all results including hyperparameters
    >>> save_to_json(results, "output/estimates.json")
    
    >>> # Save specific hyperparameters only
    >>> save_to_json(
    ...     results, "output/estimates.json",
    ...     hyperparameter_names=['tau', 'prec_o']
    ... )
    
    >>> # Save only theta and hyperparameters
    >>> save_dalia_results_to_json(
    ...     results, "output/estimates.json",
    ...     marginals_hp=marginals,
    ...     items_to_save=['theta_external', 'hyperparameters']
    ... )
    """

    # Determine which items to save (default: all)
    if items_to_save is None:
        items_to_save = [
            "theta_internal",
            "theta_external",
            "x",
            "cov_theta_internal_diagonal",
            "cov_theta_internal_full",
            "hyperparameters",
        ]

    # Build output dictionary
    dalia_estimates = {}

    # Save basic results
    if "theta_internal" in items_to_save and "theta_internal" in results:
        dalia_estimates["theta_internal"] = results["theta_internal"]

    if "theta_external" in items_to_save and "theta" in results:
        dalia_estimates["theta_external"] = results["theta"]

    if "x" in items_to_save and "x" in results:
        dalia_estimates["x"] = results["x"]

    # Save covariance matrices
    if "cov_theta_internal_diagonal" in items_to_save and "cov_theta_internal" in results:
        cov = results["cov_theta_internal"]
        diagonal = np.diag(cov) if hasattr(cov, "shape") and len(cov.shape) == 2 else cov
        dalia_estimates["cov_theta_internal_diagonal"] = diagonal

    if "cov_theta_internal_full" in items_to_save and "cov_theta_internal" in results:
        cov = results["cov_theta_internal"]
        dalia_estimates["cov_theta_internal_full"] = cov

    # Save hyperparameter distributions
    if (
        "hyperparameters" in items_to_save
        and "marginals_hp" in results
    ):
        marginals_hp = results["marginals_hp"]
        # Handle different possible structures of marginals_hp
        hp_dict = marginals_hp.get("hyperparameters", marginals_hp)

        # Determine which hyperparameters to save
        if hyperparameter_names is None:
            hyperparameter_names = list(hp_dict.keys())

        hyperparameters = {}
        for hp_name in hyperparameter_names:
            if hp_name in hp_dict:
                hp_data = hp_dict[hp_name]

                hyperparameters[hp_name] = {}

                # Save mean and variance (already floats, no conversion needed)
                if "mean_external" in hp_data:
                    hyperparameters[hp_name]["mean"] = hp_data["mean_external"]
                if "variance_external" in hp_data:
                    hyperparameters[hp_name]["variance"] = hp_data["variance_external"]

                # Save quantiles as pairs (already list structure, no conversion needed)
                if "quantiles" in hp_data and "external" in hp_data["quantiles"]:
                    quantiles = hp_data["quantiles"]["external"]
                    if "pairs" in quantiles:
                        hyperparameters[hp_name]["quantile_pairs"] = quantiles["pairs"]

                # Save PDF data as pairs: [[x1, y1], [x2, y2], ...]
                # pdf_data is [array_x, array_y], both numpy arrays that need conversion
                if "pdf_data" in hp_data:
                    pdf_x = hp_data["pdf_data"][0]  # numpy array
                    pdf_y = hp_data["pdf_data"][1]  # numpy array
                    # Convert numpy arrays to lists and zip
                    pdf_pairs = list(zip(pdf_x.tolist(), pdf_y.tolist()))
                    hyperparameters[hp_name]["pdf_pairs"] = pdf_pairs

        if hyperparameters:
            dalia_estimates["hyperparameters"] = hyperparameters

    # Create output directory if needed
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save to JSON
    with open(filename, "w") as f:
        json.dump(dalia_estimates, f, indent=2, cls=NumpyEncoder)

    print(f"Saved DALIA results to: {filename}")
