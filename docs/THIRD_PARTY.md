# Third-party notices

The root MIT license covers this project's source, numerical data and original figures. It does not relicense external dependencies.

* **SCTOP**, Chao Li (2026), MIT: https://github.com/lichaoBIT/SCTOP. The initial-field recipes and design-problem conventions in `exp/asg_sctop_reproduction_20261105/` follow this work; retain [its license](../licenses/SCTOP.txt). Our implementation uses fully integrated Q1/HEX8 elasticity with coincident design and analysis grids. It does not implement SCTOP's HEX20 sub-cell discretization, and does not claim to reproduce that discretization's speedup.
* **muTopOpt**, Lars Pastewka and contributors, MIT: https://github.com/muSpectre/muTopOpt. Download the pinned upstream revision for the optional native comparison. [License](../licenses/muTopOpt.txt).
* **muGrid**, LGPL-3.0: https://github.com/muSpectre/muGrid. Its source and binary are not vendored. `exp/asg_mutopopt_20261104/results/mugrid_cuda131.patch` changes an upstream source file and remains subject to the [muGrid license](../licenses/muGrid.txt). The patch does not change the physical operator, PCG recurrence or preconditioner mathematics.
* **PyTorch, NumPy, SciPy, pandas, Matplotlib, Ninja and CuPy** retain their respective upstream licenses. They are installed as dependencies, not relicensed by this repository.
* **NVIDIA CUDA/cuFFT** are external toolkit dependencies. No CUDA toolkit headers or libraries are included in the release.

The terms Green and Green–Jacobi describe comparison methods. Citations and mathematical attribution do not imply that these authors endorse this implementation. Journal PDFs, downloaded literature and private review files are excluded.
