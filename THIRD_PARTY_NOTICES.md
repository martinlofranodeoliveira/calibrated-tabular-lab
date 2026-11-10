# Exact dependency artifacts

MIT applies to original project source. The exact selected distributions retain their own licenses, including bundled native components. These are artifact-specific notices; they do not relicense dependencies.

| Package | Version | SHA256 |
|---|---|---|
| build | 1.6.1 | `ecd351a4be9d35a9eaaba244a7687143c9c7d4aea6ac964e7e7ddab20cbcf4e7` |
| cloudpickle | 3.1.2 | `9acb47f6afd73f60dc1df93bb801b472f05ff42fa6c84167d25cb206be1fbf4a` |
| joblib | 1.6.0 | `3dbbf9f6e4b592a2357b854608e980fe6390d131d7a82f011a377ef2ebef7aba` |
| narwhals | 2.28.0 | `1f69d6385ec30e2fa6e1fdeb37c562ecabd0f713f6f3e3b58a668c5c9df9567f` |
| numpy | 2.5.4 | `6ffa07666f8da0eef81d149934a626d0d95fbd6838432a33e66245423a9062c0` |
| packaging | 26.3 | `d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c` |
| pyproject_hooks | 1.3.3 | `5fc53fdac9f7bd63fbcdc868fb5f90b4784d78a53a3d3388cd738b807441a20b` |
| scikit-learn | 1.9.1 | `55e79d6e9b0923f1a978179822bd43d7f5543f45e970a00fe861f43486380aba` |
| scipy | 1.18.1 | `fdaf5ea890a6183d0565f51a61799d67081bd5b1cf03c5f4b3fd3732108625c9` |
| setuptools | 84.2.1 | `ca140b45d5325a9478f880e351484f62fef15f336462e449ca0c03c873d589c2` |
| threadpoolctl | 3.8.0 | `f1b49c82704d50f13ffae019c727b0fa4f6791a58f48f06a09c45face1be66da` |
| wheel | 0.48.0 | `3217dcc807155e45db462d7ef2431f5ddda0d7273b700d05a67b271ceb1287ab` |

Primary texts copied from the actual wheel distributions and selected Python/Debian base are in `third_party/licenses/`. NumPy and SciPy wheel texts include bundled OpenBLAS and compiler-runtime notices. The runtime and build closures are recorded separately in `requirements-runtime.lock` and `requirements.lock`. The public container uses the immutable official Python image digest in `Dockerfile.prepare`; its own installed base notices remain present.

No dependency is described as original project code. No external model, model download or API is used at runtime.
