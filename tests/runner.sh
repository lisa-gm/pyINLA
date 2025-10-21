
#!/bin/bash

# DALIA Test Runner Script
# Automatically detects available backends and runs appropriate tests
# Compatible with bash, zsh, tcsh, and csh shells

# =============================================================================
# Shell Detection and Compatibility Functions
# =============================================================================

detect_shell() {
    # Detect the current shell type
    if [ -n "$BASH_VERSION" ]; then
        echo "bash"
    elif [ -n "$ZSH_VERSION" ]; then
        echo "zsh"
    elif [ -n "$tcsh" ]; then
        echo "tcsh"
    elif [ -n "$version" ]; then
        echo "tcsh"  # tcsh sets $version variable
    else
        # Fallback: check $SHELL variable
        case "$SHELL" in
            *bash*) echo "bash" ;;
            *zsh*) echo "zsh" ;;
            *tcsh*) echo "tcsh" ;;
            *csh*) echo "csh" ;;
            *) echo "unknown" ;;
        esac
    fi
}

set_env_var() {
    # Set environment variable using appropriate shell syntax
    local var_name="$1"
    local var_value="$2"
    local shell_type=$(detect_shell)
    
    case "$shell_type" in
        bash|zsh|sh)
            export "$var_name=$var_value"
            ;;
        tcsh|csh)
            setenv "$var_name" "$var_value"
            ;;
        *)
            # Try both methods as fallback
            export "$var_name=$var_value" 2>/dev/null || setenv "$var_name" "$var_value" 2>/dev/null
            ;;
    esac
}

print_message() {
    # Print formatted message with consistent styling
    local message="$1"
    local type="$2"  # INFO, SUCCESS, WARNING, ERROR
    
    case "$type" in
        SUCCESS) echo "✓ $message" ;;
        WARNING) echo "⚠ $message" ;;
        ERROR) echo "✗ $message" ;;
        *) echo "• $message" ;;
    esac
}

# =============================================================================
# Backend Detection Functions
# =============================================================================

check_gpu_availability() {
    # Check if NVIDIA GPU is available via nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
            print_message "GPU detected: $gpu_count NVIDIA GPU(s) available" "SUCCESS"
            return 0
        else
            print_message "nvidia-smi found but not working properly" "WARNING"
            return 1
        fi
    else
        print_message "nvidia-smi not found - no NVIDIA GPU detected" "INFO"
        return 1
    fi
}

check_cupy_installation() {
    # Check if CuPy is installed and working in the current environment
    print_message "Checking CuPy installation and functionality..." "INFO"
    
    # Try to import and test CuPy
    python -c "
import sys
try:
    import cupy as cp
    # Test basic CUDA operation
    test_array = cp.array([1, 2, 3])
    result = cp.sum(test_array)
    print('CuPy test successful: sum([1,2,3]) =', result.get())
    sys.exit(0)
except ImportError:
    print('CuPy not installed')
    sys.exit(1)
except Exception as e:
    print('CuPy installed but not working:', str(e))
    sys.exit(2)
" 2>/dev/null
    
    local cupy_status=$?
    case $cupy_status in
        0)
            print_message "CuPy is installed and working correctly" "SUCCESS"
            return 0
            ;;
        1)
            print_message "CuPy is not installed" "WARNING"
            return 1
            ;;
        2)
            print_message "CuPy is installed but not functioning (check CUDA drivers)" "WARNING"
            return 1
            ;;
        *)
            print_message "Unable to test CuPy installation" "ERROR"
            return 1
            ;;
    esac
}

determine_available_backends() {
    # Determine which backends are available for testing
    local backends=""
    
    print_message "Detecting available backends..." "INFO"
    echo
    
    # NumPy is always available
    print_message "NumPy backend: Always available" "SUCCESS"
    backends="numpy"
    
    # Check for GPU + CuPy
    if check_gpu_availability; then
        if check_cupy_installation; then
            print_message "CuPy backend: Available" "SUCCESS"
            backends="$backends cupy"
        else
            print_message "CuPy backend: Unavailable (CuPy not working)" "WARNING"
        fi
    else
        print_message "CuPy backend: Unavailable (No GPU detected)" "WARNING"
    fi
    
    echo "$backends"
}

# =============================================================================
# Test Execution Functions
# =============================================================================

choose_tests_to_run() {
    # Display which tests will be run based on available backends
    local backends="$1"
    
    echo "Test execution plan:"
    
    if echo "$backends" | grep -q "numpy"; then
        echo "  - CPU backend (NumPy)"
    fi
    
    if echo "$backends" | grep -q "cupy"; then
        echo "  - GPU backend (CuPy)"
    fi
}

run_numpy_tests() {
    # Run tests with NumPy backend
    echo "Running testing suite on CPU backend (NumPy)..."
    echo "==============================================="
    
    set_env_var "ARRAY_MODULE" "numpy"
    
    # Run pytest
    if command -v pytest >/dev/null 2>&1; then
        pytest . -v
        local exit_code=$?
    else
        print_message "pytest not found - cannot run tests" "ERROR"
        return 1
    fi
}

run_cupy_tests() {
    # Run tests with CuPy backend
    echo "Running testing suite on GPU backend (CuPy)..."
    echo "==============================================="
    
    set_env_var "ARRAY_MODULE" "cupy"
    
    # Run pytest
    if command -v pytest >/dev/null 2>&1; then
        pytest . -v
        local exit_code=$?
    else
        print_message "pytest not found - cannot run tests" "ERROR"
        return 1
    fi
}

# =============================================================================
# Main Execution Logic
# =============================================================================

main() {
    # Main function that orchestrates the test execution
    local shell_type=$(detect_shell)
    
    echo "=============================================="
    echo "             DALIA Tests Runner"
    echo "=============================================="
    echo "Detected shell: $shell_type"
    echo "Current directory: $(pwd)"
    echo
    
    # Detect available backends
    local available_backends=$(determine_available_backends)
    
    # Show test plan
    choose_tests_to_run "$available_backends"
    
    # Ask for confirmation
    echo -n "Proceed with test execution? [y/N]: "
    read confirmation
    case "$confirmation" in
        [yY]|[yY][eE][sS])
            echo "Starting tests..."
            ;;
        *)
            echo "Test execution cancelled by user."
            exit 0
            ;;
    esac
    
    echo
    local overall_success=0
    
    # Run NumPy tests (always available)
    if echo "$available_backends" | grep -q "numpy"; then
        if ! run_numpy_tests; then
            overall_success=1
        fi
    fi
    
    # Run CuPy tests if available
    if echo "$available_backends" | grep -q "cupy"; then
        if ! run_cupy_tests; then
            overall_success=1
        fi
    fi
    
    # Final summary
    echo "=============================================="
    if [ $overall_success -eq 0 ]; then
        echo "     All tests completed successfully!"
    else
        echo "  Some tests failed - check output above"
    fi
    echo "=============================================="
    
    exit $overall_success
}

# =============================================================================
# Script Entry Point
# =============================================================================

# Check if script is being sourced or executed
if [ "${BASH_SOURCE[0]}" = "${0}" ] 2>/dev/null || [ "${(%):-%x}" = "${0}" ] 2>/dev/null; then
    # Script is being executed directly
    main "$@"
fi

