import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def plot_marginal_distributions_hp(marginals_hp):
    """Plot marginal distributions of hyperparameters in both internal and external parametrizations."""

    # Get all hyperparameters
    hyperparams = marginals_hp['hyperparameters']
    n_params = len(hyperparams)
    
    # Create subplot grid: n_params rows, 2 columns (internal left, external right)
    fig, axes = plt.subplots(n_params, 2, figsize=(15, 5*n_params))
    
    # Handle case of single parameter
    if n_params == 1:
        axes = axes.reshape(1, -1)
    
    # Quantile colors and labels
    colors = ['#DEB887', '#DEB887','darkred', '#DEB887', '#DEB887']
    labels = ['2.5%', '25%', '50%', '75%','97.5%']
    
    for row, (param_name, param_data) in enumerate(hyperparams.items()):
        # Get internal parameters
        mean_internal = param_data['mean_internal']
        var_internal = param_data['variance_internal']
        std_internal = np.sqrt(var_internal)
        
        # Get external parameters  
        mean_external = param_data['mean_external']
        var_external = param_data['variance_external']
        theta_external, pdf_external = param_data['pdf_data']
        
        # Get quantiles
        quantile_pairs_internal = param_data['quantiles']['internal']['pairs']
        quantile_pairs_external = param_data['quantiles']['external']['pairs']
        
        # ===== LEFT PLOT: INTERNAL PARAMETRIZATION =====
        ax_left = axes[row, 0]
        
        # Create internal distribution (Gaussian)
        x_internal = np.linspace(mean_internal - 4*std_internal, mean_internal + 4*std_internal, 100)
        pdf_internal = norm.pdf(x_internal, loc=mean_internal, scale=std_internal)
        
        print("x_internal: ", x_internal[:10])
        print("x_external: ", theta_external[:10])
        print("diff (x internal - external): ", np.linalg.norm(x_internal - theta_external))
        print("pdf_internal: ", pdf_internal[:10])
        print("pdf_external: ", pdf_external[:10])
        print("diff pdf internal - external: ", np.linalg.norm(pdf_internal - pdf_external))
        # Plot internal PDF
        ax_left.plot(x_internal, pdf_internal, 'b-', linewidth=2, label='PDF (Internal)')
        
        # Mark internal mean
        ax_left.axvline(mean_internal, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean = {mean_internal:.3f}')
        
        # Mark internal quantiles
        for i, (prob, q_val) in enumerate(quantile_pairs_internal):
            if i < len(labels):
                ax_left.axvline(q_val, color=colors[i], linestyle=':', linewidth=2,
                               label=f'{labels[i]} = {q_val:.3f}')
        
        ax_left.set_xlabel(f'{param_name} (internal scale)')
        ax_left.set_ylabel('PDF')
        ax_left.set_title(f'{param_name}: Internal Distribution (Gaussian)')
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)
        
        # ===== RIGHT PLOT: EXTERNAL PARAMETRIZATION =====
        ax_right = axes[row, 1]
        
        # Plot external PDF
        ax_right.plot(theta_external, pdf_external, 'b-', linewidth=2, label='PDF')
        
        # Mark external mean
        ax_right.axvline(mean_external, color='red', linestyle='--', linewidth=2,
                        label=f'Mean = {mean_external:.3f}')
        
        # Mark external quantiles
        for i, (prob, q_val) in enumerate(quantile_pairs_external):
            if i < len(labels):
                ax_right.axvline(q_val, color=colors[i], linestyle=':', linewidth=2,
                                label=f'{labels[i]} = {q_val:.3f}')
        
        ax_right.set_xlabel(f'{param_name} ')
        ax_right.set_ylabel('PDF')
        ax_right.set_title(f'{param_name}: Marginal Distribution')
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    # plt.show()
    
    return fig, axes
