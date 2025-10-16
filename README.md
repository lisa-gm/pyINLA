

# DALIA
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)  

---

Python implementation of the methodology of integrated nested Laplace approximations (INLA), putting the accent on portability, modularity and performance.

If you want to get involved in the development of DALIA, please feel free to contact us directly.

## Installation
Detailed installation instructions are provided in [install.md](./install.md).

# Tests, Examples and Benchmarks
## Testing
TODO

## Examples

Some examples are provided with running scripts. The examples are being tracked using `git-lfs`, to download them, run the following commands:
```bash
git lfs pull
git lfs checkout
```

You can then navigate in the `examples/` directory and run the given examples. For example, to run a Gaussian Spatio-Temporal model, you can run:
```bash
python gst_small/run.py
```

## Benchmarks
TODO

## Known Installation Issues
The `sqlite` module might not work properly. Forcing the following version of `sqlite` might help:
```bash
conda install conda-forge::sqlite=3.45.3
```

# Citing DALIA

The main DALIA paper describing its high performance computing strategies is available through the following reference:

``` bibtex
@misc{gaedkemerzhäuser2025acceleratedspatiotemporalbayesianmodeling,
      title={Accelerated Spatio-Temporal Bayesian Modeling for Multivariate Gaussian Processes}, 
      author={Lisa Gaedke-Merzhäuser and Vincent Maillou and Fernando Rodriguez Avellaneda and Olaf Schenk and Mathieu Luisier and Paula Moraga and Alexandros Nikolaos Ziogas and Håvard Rue},
      year={2025},
      eprint={2507.06938},
      archivePrefix={arXiv},
      primaryClass={stat.CO},
      url={https://arxiv.org/abs/2507.06938}, 
}
```