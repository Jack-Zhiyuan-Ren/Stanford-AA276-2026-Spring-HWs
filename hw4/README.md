# Stanford AA276: Principles of Safety-Critical Autonomy, Spring 2026

## Homework 4

In this homework assignment, you will gain experience designing controllers to compete in the cart-pole race!

## Instructions
Follow the instructions provided in the Homework 4 handout.

## Environment Setup

Create a virtual environment with `python -m venv [name]`<br>
Activate the virtual environment with `source [name]/bin/activate` (Linux)<br>
Install required packages with `pip install -r requirements.txt`

## Submission

Submit your work to Gradescope. The autograder runs all coding problems and posts a score plus a class leaderboard for the bonus problem.

### Files to upload

Upload these files **directly at the submission root** (no nested folders):

- `problem_1_3.py` — must define class `Controller` (see template).
- `problem_2_1.py` — must define classes `ControllerA` and `ControllerB`.
- `problem_2_2.py` — must define class `Controller`.

You may also upload any helper `.py` modules that your controllers import (e.g. `my_lqr.py`, `mpc_utils.py`). Do **not** upload `utils/`, `scripts/`, `requirements.txt` — those live on the grading server already.

### Leaderboard

The autograder reports `Avg Distance 2.2 (m)` to the class leaderboard. Per the handout's bonus rule, your leaderboard entry only counts as "qualified" if **all 5** of your Problem 2.2 runs survive the full 10 s. If any run falls, your leaderboard distance is reported as 0.

### Available packages

The grading environment ships:

- `numpy`, `scipy`, `matplotlib`, `tqdm`
- `jax` (CPU-only build — no GPU on the grader)
- `cvxpy` (with default solvers OSQP, ECOS, SCS)
- `hj_reachability`

**Not available**: `torch`. If you want to use PyTorch, you must precompute results locally and ship them as data files (`.npy`, `.pkl`) alongside your `.py` files.

Note that any value-function or policy that you compute inside `Controller.__init__` will be re-computed on every grading run (the autograder instantiates each controller class fresh). 