# CAPOF — Context-Adaptive Policy Framework

This repository contains the open-source code for our paper on robust and reactive robotic manipulation via uncertainty-aware imitation learning:

**A context-adaptive policy framework for robust and reactive robotic manipulation via uncertainty-aware imitation learning**

[RAS paper](https://doi.org/10.1016/j.robot.2026.105649) | [ArXiv paper](https://arxiv.org/abs/2410.24035) | [ELIB paper](https://elib.dlr.de/225946/) | [YouTube](https://youtu.be/vxQcqhXqHA8)

**Authors:** Tim R. Winter, Leonard Klüpfel, Ashok M. Sundaram, Werner Friedl, Maximo A. Roa, Freek Stulp, João Silvério

Responsible: Tim R. Winter (tim.winter@dlr.de)
Research Scientist @ German Aerospace Center (DLR), Institute of Robotics and Mechatronics, Munich, Germany

## 💡 Overview

CAPOF is a context-adaptive framework that leverages the epistemic uncertainty captured from demonstrations to enable robust and reactive robotic manipulation. It combines a Gaussian-process-based learning-from-demonstration (LfD) policy with two supplementary policies in a Mixture-of-Experts (MoE) formulation:

- **LfD policy** — reproduces the demonstrated behavior from the robot state and low-dimensional task-dependent parameters describing the environment or task context.
- **Stabilizing policy** — uses the GP's epistemic uncertainty to steer the robot towards low-uncertainty regions (close to the demonstrated data), improving robustness to out-of-distribution states. Derived from the gradient of the GP predictive variance.
- **Goal attractor policy** — improves convergence by steering the robot towards a demonstrated goal pose selected by its kernel activation at the current state; its action follows the gradient of that kernel activation.

The three experts are combined with state-dependent mixing coefficients, so the dominant policy changes with the current state and uncertainty — unifying reactive imitation, out-of-distribution robustness, and convergence towards demonstrated goals.

The repository ships with three runnable examples: a self-contained 6D motion reproduction, a LASA handwriting example, and two robomimic *Lift* manipulation use-cases (force-conditioned and object-centric).

## 🗂️ Repository structure

```
capof/                    # Core framework (pure Python, no simulation deps)
├── capof.py                # ContextAdaptivePolicyFramework
├── gp_scikit.py            # GP with epistemic uncertainty + gradients
├── outer_control_loop.py   # Base execution loop (template-method pattern)
└── training.py             # Shared GP fitting helper (train_capof)

examples/                 # Runnable examples (each a self-contained folder)
├── robomim/                # Robomimic Lift use-cases
│   ├── force_conditioned.py  # Entry script: policy conditioned on contact forces
│   ├── object_centric.py     # Entry script: grasp + lift skills, random cube pose
│   ├── robomim_env.py        # Gymnasium wrapper for robomimic environments
│   ├── robomim_interface.py  # Control loops (base + two use-case subclasses)
│   ├── data_processing.py    # Demo loading + (X, Y) feature construction
│   ├── live_viewer_utils.py  # Live viewer aids (keys, stuck monitor, live plot)
│   ├── utils.py              # Small shared helpers (video, controller bounds)
│   ├── evaluate.sh           # Batch evaluation with success-rate statistics
│   ├── convert_dataset.py    # Converts the robomimic dataset into the CAPOF dataset
│   └── datasets/             # Demonstration datasets (generated, git-ignored)
├── lasa/                   # LASA handwriting example
│   ├── lasa.py               # Entry script
│   ├── lasa_interface.py     # Control loop + vector-field visualization
│   ├── lasa_data.py          # .mat loading + dataset location check
│   └── dataset/              # LASA dataset (cloned from its own repo, git-ignored)
└── 6d_example/             # Self-contained 6D motion reproduction example

bbo/                      # Black-box hyperparameter optimization (CMA-ES / BO)
tests/                    # Unit tests for the capof core
```

## 📦 Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/DLR-RM/CAPOF.git
    cd CAPOF
    ```

2. Create a Python 3.12 environment and install the package with all extras:

    ```bash
    conda create -n capof python=3.12
    conda activate capof
    pip install -e ".[dev]"
    ```

    This installs the `capof` package in editable mode, together with all
    runtime dependencies and the development tools for linting, type
    checking, and testing (`dev`). Alternatively, the requirements files
    can still be used: `pip install -r requirements.txt` and
    `pip install -r requirements-dev.txt`.

3. (Only for the LASA example) Clone the LASA handwriting dataset. It is not
   included in this repository (it is free for non-commercial academic use
   only, see its [repository](https://github.com/epfl-lasa/LASAHandwritingDataset)):

    ```bash
    git clone https://github.com/epfl-lasa/LASAHandwritingDataset.git examples/lasa/dataset
    ```

4. (Only for the robomimic examples) The *Lift* demonstration dataset is not
   included in this repository. It is generated from the official robomimic
   v1.5 *Lift* (PH model) low-dim dataset
   ([`low_dim_v15.hdf5`](https://huggingface.co/datasets/robomimic/robomimic_datasets/tree/main/v1.5/lift/ph))
   by replaying each demo in the robosuite simulation and appending the
   gripper–cube contact forces / grasp status used by the examples:

    ```bash
    python examples/robomim/convert_dataset.py path/to/low_dim_v15.hdf5
    ```

    The script writes `examples/robomim/datasets/lift_ph_v15.hdf5`
    (git-ignored). It takes a few minutes (200 demos are replayed in the
    simulation). On headless machines set `MUJOCO_GL=egl` so that MuJoCo can
    create its offscreen rendering context.

## 🤖 Examples

Run all examples from the repository root.

### 6D motion reproduction

A self-contained example: a helical motion is generated, a GP is trained on the resulting (pose, twist) pairs, and CAPOF reproduces the motion from an offset start pose.

```bash
python examples/6d_example/six_dimensional_example.py
```

### LASA handwriting

Runs CAPOF on the LASA handwriting dataset from random start points and reports success rate, iteration count, and mean distance to the demonstrations per letter. This example is the same as the one presented in Section 5.2 of the [paper](https://doi.org/10.1016/j.robot.2026.105649).

The LASA dataset must be cloned first (see step 3 in [Installation](#-installation)); pass `--lasa-repo` to use a different location.

```bash
python examples/lasa/lasa.py                        # animated visualization per start point
python examples/lasa/lasa.py --mode headless        # no plots (fast)
python examples/lasa/lasa.py --letters Angle Line   # run a subset of letters
python examples/lasa/lasa.py --help                 # all options (hyperparameters, ...)
```

### Robomimic Lift (force-conditioned)

The policy input is conditioned on the two gripper–cube contact forces; the cube starts at a fixed pose. This use case is set up to be as similar as possible to the real-robot experiment described in Section 5.3 of the [paper](https://doi.org/10.1016/j.robot.2026.105649).

The demonstration dataset must be generated first (see step 4 in [Installation](#-installation)); pass `--dataset` to use a different location.

```bash
python examples/robomim/force_conditioned.py                 # headless, video saved to examples/robomim/results/
python examples/robomim/force_conditioned.py --live-render   # interactive MuJoCo viewer
python examples/robomim/force_conditioned.py --seed 1       # fixed environment seed
```

### Robomimic Lift (object-centric)

Two skills (grasp + lift) are executed from a single context: the grasp skill runs in a symmetry-canonicalized object-centric frame that tracks the cube, and the lift skill in the world frame. The cube starts at a random pose. This use case is close to the real-robot experiment in Section 5.6 of the [paper](https://doi.org/10.1016/j.robot.2026.105649), with the phases matching the provided robomimic dataset.

The demonstration dataset must be generated first (see step 4 in [Installation](#-installation)); pass `--dataset` to use a different location.

```bash
python examples/robomim/object_centric.py                 # headless, video saved to examples/robomim/results/
python examples/robomim/object_centric.py --live-render   # interactive MuJoCo viewer
python examples/robomim/object_centric.py --seed 1        # fixed environment seed
```

#### Live viewer

With `--live-render` a MuJoCo passive viewer window opens (requires an X display). A live plot in the corner shows the task parameters over simulation time (contact forces for force-conditioned, grasp status for object-centric). Keys:

- `space` — re-place the cube at its initial pose (the plot trace continues)
- `r` — restart the episode with a fresh random seed (the plot trace resets)
- close the window — stop the episode

#### Batch evaluation

`examples/robomim/evaluate.sh` runs N trials (fresh random seed each) and reports the success rate:

```bash
cd examples/robomim
./evaluate.sh force_conditioned 20
./evaluate.sh object_centric 10
```

## ⚙️ Hyperparameter optimization (BBO)

The GP/CAPOF hyperparameters of all examples can be optimized with CMA-ES or Bayesian optimization. The black-box functions share the exact data pipeline and training code of the entry scripts, so the printed "paste-ready" hyperparameters can be dropped straight into the corresponding entry script (via its command-line flags).

```bash
python bbo/run_bbo.py --task lasa --method cmaes --n-trials 50
python bbo/run_bbo.py --task robomim_force_conditioned --method bayes --n-trials 30
python bbo/run_bbo.py --task robomim_object_centric --method cmaes --n-trials 30
```

The `lasa` task requires the cloned LASA dataset, and the `robomim_*` tasks require the generated robomimic datasets (see steps 3 and 4 in [Installation](#-installation)).

Re-running the same command with the same `--output-dir` resumes from the last completed trial. Results and plots:

```bash
python bbo/analyze_bbo_results.py --results-dir bbo/bbo_optimization_results_lasa --plot
```

## 🧑‍💻 Development

The Makefile is the primary workflow:

```bash
make lint          # ruff (hard errors + warnings)
make mypy          # type checking
make pytest        # unit tests (capof core)
make pytest_cov    # unit tests with coverage report
make format        # isort + black (line length 127)
make commit-checks # format + mypy
```

The unit tests cover the `capof` core (GP epistemic queries, policy combination, control-loop mechanics) and run without any simulation dependencies.

## 📝 Notes

- The robomim examples require the robomimic/robosuite stack (see `requirements.txt`); on first use robosuite may download robot models.
- The live viewer requires an X display and `pynput` (keyboard listener).
- Videos are written with OpenCV (`mp4v` codec) to `examples/robomim/results/<use_case>/<task>/<seed>.mp4`.

## 📚 Citation

If you use our ideas in a research project or publication, please cite as follows:

```bibtex
@article{WINTER2026105649,
  author = {Tim R. Winter and Leonard Klüpfel and Ashok M. Sundaram and Werner Friedl and Maximo A. Roa and Freek Stulp and João Silvério},
  title = {A context-adaptive policy framework for robust and reactive robotic manipulation via uncertainty-aware imitation learning},
  journal = {Robotics and Autonomous Systems},
  volume = {205},
  pages = {105649},
  year = {2026},
  issn = {0921-8890},
  doi = {https://doi.org/10.1016/j.robot.2026.105649},
  keywords = {Imitation learning, Learning from demonstration, Machine learning for robot control, Robotic manipulation, Dynamical systems, Uncertainty awareness},
}
```

## 🏷️ License

This repository is released under the MIT license. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgements

This work builds on the following open-source projects and datasets:

- [Robomimic](https://github.com/ARISE-Initiative/robomimic) and [Robosuite](https://github.com/robosuite/robosuite) — the manipulation benchmark and *Lift* task used by the robomim examples.
- [Robomimic Datasets](https://huggingface.co/datasets/robomimic/robomimic_datasets) (HuggingFace) — the official *Lift* (PH) v1.5 demonstration dataset from which the CAPOF robomimic datasets are generated (not distributed with this repository).
- [LASA](https://github.com/epfl-lasa/LASAHandwritingDataset) - the handwriting dataset (free for non-commercial academic use only; not distributed with this repository).
- [scikit-learn](https://scikit-learn.org/) — the `GaussianProcessRegressor` that our GP implementation extends.
