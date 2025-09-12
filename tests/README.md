# DALIA Testing Framework

This directory contains the comprehensive test suite for the DALIA framework, organized according to the **testing pyramid** principle.

## 1. **Test Architecture Overview**

```
integration/                ← Few, expensive, end-to-end tests
component_integration/      ← Medium, component interaction tests
unit/                       ← Many, fast, isolated tests
```

## 2. **Testing Suite Structure**

### 2.1 [`unit/`](unit/README.md) - **Fast, Isolated Tests**
- **Purpose**: Test individual components in isolation
- **Speed**: < 100ms per test (typically)
- **Dependencies**: Minimal (mocked when needed)
- **Quantity**: Many (hundreds to thousands)

**Examples**: Configuration validation, utility functions, error handling

#### Add to `unit/` when:
- Testing pure logic without external dependencies
- Validating configuration and input parsing
- Testing error conditions and edge cases
- Verifying utility functions and calculations


### 2.2 [`component_integration/`](component_integration/README.md) - **Component Interaction Tests**  
- **Purpose**: Test interactions within or between closely related components
- **Speed**: 100ms - 5s per test
- **Dependencies**: Real system resources (MPI, GPU, files)
- **Quantity**: Medium (dozens to hundreds)

**Examples**: MPI communication, GPU kernels, backend compatibility

**Note**: These tests need workflows specific to testing cluster with enabled GPU and multiprocessing (multi-node) capabilities.

#### Add to `component_integration/` when:
- Testing real MPI communication
- Testing actual GPU operations
- Testing file I/O and data persistence
- Testing cross-backend compatibility
- Testing hardware detection and initialization


### 2.3 [`integration/`](integration/README.md) - **End-to-End Workflow Tests**
- **Purpose**: Test complete user workflows and multi-component interactions
- **Speed**: Minutes to hours per test
- **Dependencies**: Full system setup + datasets
- **Quantity**: Few (tens)

**Examples**: Complete Gaussian Process workflows, performance benchmarks, scientific accuracy

**Note**: These tests need workflows specific to testing cluster with enabled GPU and multiprocessing (multi-node) capabilities.

#### Add to `integration/` when:
- Testing complete user workflows
- Validating scientific accuracy end-to-end
- Testing performance and scalability
- Verifying multiple components working together
- Testing example code and documentation



## 3. **Running Tests**

### Quick Development Testing
```bash
# Run fast unit tests only
python -m pytest tests/unit/ -v

# Run tests for specific component
python -m pytest tests/unit/communicator/ -v
```

### Comprehensive Testing
```bash
# Run all tests (slow!)
python -m pytest tests/ -v

# Run unit + component integration (skip slow integration)
python -m pytest tests/unit/ tests/component_integration/ -v

# Run with coverage
python -m pytest tests/unit/ --cov=src/dalia --cov-report=html
```

### CI/CD Testing Pipeline

Todo: ...

## 4. **Testing Guidelines**

1. **Follow the pyramid**: More unit tests, fewer integration tests
2. **Start with unit tests** when adding new functionality
3. **Mock external dependencies** in unit tests
4. **Use real dependencies** in component/integration tests
5. **Skip gracefully** when hardware/software unavailable:
   ```python
   @pytest.mark.skipif(not backend_flags["mpi_avail"], reason="MPI not available")
   ```
6. **Clean up resources** (files, GPU memory, MPI communicators)
7. **Use descriptive test names** that explain what's being tested


### 4.1 **Contributing New Tests**

When adding new functionality:

1. **Start with unit tests** in `tests/unit/[component]/`
2. **Add component integration tests** if using real system resources
3. **Consider integration tests** for user-facing workflows
4. **Update relevant README files** when adding new test categories
5. **Use appropriate pytest markers** for test categorization

### 4.2 **Example Component Test Structure**

Here is a suggested test structure for a hypothetical DALIA new component `new_component`:

```
tests/
├── unit/
│   └── new_component/
│       ├── conftest.py
│       ├── test_basic.py
│       └── test_config.py
│       
├── component_integration/
│   └── new_component/
│       ├── conftest.py
│       ├── test_hardware_integration.py
│       └── test_cross_backend.py
│
└── integration/
    └── test_new_component_workflows.py
```

In the case of the implementation of a new feature to an existing component, add/adapt tests to the existing component test directories.

## 5. **Test Configuration**

### Pytest Configuration (`pyproject.toml` or `pytest.ini`)
```ini
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "slow: marks tests as slow (> 30 seconds)",
    "gpu_required: marks tests that require GPU",
    "multi_node: marks tests that require multiple nodes",
    "large_memory: marks tests that require > 16GB RAM"
]
```

### Environment Variables
```bash
export MPI_CUDA_AWARE=1    # Enable CUDA-aware MPI testing
export USE_NCCL=1          # Enable NCCL testing  
export ARRAY_MODULE=cupy   # Use CuPy for GPU tests
```


