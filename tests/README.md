# DALIA testing folder
The DALIA testing suite is in construction. Integration tests are not yet available, for now you can refer to the examples provided in the `examples/` directory for testing of the entire pipeline.


## How to run tests

The tests can either be run directly using `pytest` or through the provided `runner.sh` script. The `runner.sh` script allows for more convenient selection of test categories and backends. You can either run it directly using `./runner.sh` or check the available options with `./runner.sh --help`.


## Tests status

| Reference | Status | Reason |
| --------- | ------ | ------ |
| `component_integration/solvers/sparse_solvers/sequential/test_selected_inversion()` | Not Implemented | Not Implemented                                                 |
| `component_integration/solvers/sparse_solvers/sequential/test_factorize()` | Limited (canno't check for numerical corectness) | LU decomposition instead of Cholesky due to `scipy` limitations |
| `component_integration/solvers/structured_solvers/distributed/test_factorize()` | Limited (canno't check for numerical corectness) | Distributed factorization is not numerically equal to sequential reference |

