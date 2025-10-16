#!/bin/bash

alps_install_uenv() {
    # Documentation: https://docs.cscs.ch/software/uenv/
    echo "alps_install_uenv: setting up ALPS uenv environment."

    if uenv image list | grep -q "prgenv-gnu/24.11:v1"; then
        echo "   prgenv-gnu/24.11:v1 image already exists, skipping pull"
    else
        echo "   Pulling prgenv-gnu/24.11:v1 image..."
        uenv image pull prgenv-gnu/24.11:v1 || {
            echo "❌ Error: Failed to pull prgenv-gnu/24.11:v1 image"
            return 1
        }
    fi
}

alps_start_uenv() {
    # Documentation: https://docs.cscs.ch/software/uenv/
    echo "alps_start_uenv: starting ALPS uenv environment."
    
    # Stop any existing uenv session
    uenv stop 2>/dev/null || echo "   (No existing uenv to stop)"
    
    # Start new uenv session
    echo "   Starting uenv with prgenv-gnu/24.11:v1..."
    echo "   WARNING: This is gonna start a new shell session, if you want to"
    echo "   use other functions from this script, you need to source it again."
    uenv start --view=modules prgenv-gnu/24.11:v1
}

# Function to load ALPS modules with error handling
alps_load_modules() {
    echo "alps_load_modules: loading ALPS system modules."
    
    # Check if we're already in a uenv session
    if ! uenv status &>/dev/null; then
        echo "Error: Not in a uenv session. Modules can only be loaded within a uenv session."
        return 1
    fi
    
    # Purge any existing modules
    module purge 2>/dev/null
    
    # Load required modules (excluding python to avoid conflicts with conda)
    module load cuda gcc meson ninja nccl cray-mpich cmake openblas aws-ofi-nccl netlib-scalapack || {
        echo "   Error: Failed to load required modules; cuda, gcc, meson, ninja, nccl, cray-mpich, cmake, openblas, aws-ofi-nccl, netlib-scalapack"
        echo "   Note: python/3.12.5 module excluded to preserve conda environment"
        echo "   Available modules:"
        module avail 2>&1 
        return 1
    }

    # NCCL environment setup
    export NCCL_ROOT=/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd
    export NCCL_LIB_DIR=$NCCL_ROOT/lib
    export NCCL_INCLUDE_DIR=$NCCL_ROOT/include

    # CUDA environment setup
    if [[ -z "$CUDA_HOME" ]]; then
        # CUDA_HOME not set
        echo "   Error: CUDA_HOME not set, please ensure the CUDA module properly sets CUDA_HOME or manually set CUDA_HOME to your CUDA installation directory"
        return 1
    fi
    
    export CUDA_DIR=$CUDA_HOME
    export CUDA_PATH=$CUDA_HOME
    export CPATH=$CUDA_HOME/include:$CPATH
    export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    export CPATH=$NCCL_ROOT/include:$CPATH
    export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
    export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

    return 0
}

alps_activate_conda_env() {
    echo "alps_activate_conda_env: activating DALIA conda environment."
    conda deactivate

    # Define available environments in order of preference (most performant first)
    local env_priorities=("dalia_xccl_daint" "dalia_ampi_daint" "dalia_hmpi_daint" "dalia_base_daint")
    local env_name=""
    
    # If environment name is provided as argument, use it directly
    if [[ -n "$1" ]]; then
        env_name="$1"
        echo "   Using specified conda environment '${env_name}'..."
    else
        # Check which environments are available and select the most performant one
        echo "   Checking available DALIA conda environments..."
        local available_envs=$(conda env list | grep -E "dalia_(xccl|ampi|hmpi|base)_daint" | awk '{print $1}')
        
        for preferred_env in "${env_priorities[@]}"; do
            if echo "$available_envs" | grep -q "^${preferred_env}$"; then
                env_name="$preferred_env"
                echo "   Selected most performant available environment: '${env_name}'"
                break
            fi
        done
        
        # Fallback if no DALIA environments found
        if [[ -z "$env_name" ]]; then
            echo "Warning: No DALIA-specific environments found. Available environments:"
            conda env list
            echo "   Falling back to 'base' environment..."
            env_name="base"
        fi
    fi

    echo "   Activating conda environment '${env_name}'..."
    conda activate ${env_name} || {
        echo "Error: Failed to activate conda environment '${env_name}'"
        return 1
    }
    
    # Ensure conda Python takes precedence over system modules
    echo "   Ensuring conda Python takes precedence..."
    export PATH="$CONDA_PREFIX/bin:$PATH"
    
    # Verify the correct Python is being used
    local python_path=$(which python)
    if [[ "$python_path" == *"$CONDA_PREFIX"* ]]; then
        echo "Conda environment '${env_name}' activated correctly"
        echo "   Python: $python_path"
    else
        echo "Warning: System Python may still take precedence over conda Python"
        echo "   Current Python: $python_path"
        echo "   Expected: $CONDA_PREFIX/bin/python"
    fi
    echo ""
    
    return 0
}

alps_set_perfenv() {
    echo "alps_set_perfenv: setting performance environment variables for ALPS."
    set -e
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
    export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
    export MPICH_GPU_SUPPORT_ENABLED=1

    # NCCL Performance Configuration
    # More can be found: https://docs.cscs.ch/software/communication/nccl/#using-nccl
    export NCCL_NET='AWS Libfabric'
    export NCCL_NET_GDR_LEVEL=PHB
    export NCCL_CROSS_NIC=1

    export FI_CXI_DEFAULT_CQ_SIZE=131072
    export FI_CXI_DEFAULT_TX_SIZE=32768
    export FI_CXI_DISABLE_HOST_REGISTER=1
    export FI_CXI_RX_MATCH_MODE=software
    export FI_MR_CACHE_MONITOR=userfaultfd
}

