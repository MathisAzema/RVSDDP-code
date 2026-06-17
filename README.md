* **`src/`**
  Source code of the `RVSDDP` Julia package.

* **`notebook_toy.ipynb`**
  A good entry point to learn how to use the `RVSDDP` package on a small toy
  example. In particular, useful parameters of the `train` function include:

  * `parallel`: number of forward passes
  * `time_limit` or `cut_limit`: stopping criterion
  * `shift_function`: `RVSDDP.no_shift` (`delta = 0`) or
    `RVSDDP.shift_update_random_forward` for random shifts
  * `refine_mode`: `0` for one cut at each point of the forward pass, or `1`
    to use the Shapiro-style refinement

* **`results_msppy/`**, **`results_toy/`**, **`results msppy.ipynb`** and
  **`results_toy.ipynb`**
  Scripts, notebooks and figures used to process the numerical data for the
  article.

* **`run_*.jl`**
  Scripts used to generate the numerical results for the paper.
