"""
AA 276 Homework 1 | Coding Portion | Part 2 of 3


OVERVIEW

In this file, you will implement functions for simulating the
13D quadrotor system discretely and computing the CBF-QP controller.


INSTRUCTIONS

Make sure you pass the tests for Part 1 before you begin.
Please refer to the Homework 1 handout for instructions and implementation details.

Function headers are provided below.
Your code should go into the sections marked by "YOUR CODE HERE"

When you are done, you can sanity check your code (locally) by running `python scripts/check2.py`.
"""


import torch
from part1 import f, g


"""Note: the following functions operate on batched inputs."""


def euler_step(x, u, dt):
    """
    Return the next states xn obtained after a discrete Euler step
    for states x, controls u, and time step dt.
    Hint: we have imported f(x) and g(x) from Part 1 for you to use.
    
    args:
        x: torch float32 tensor with shape [batch_size, 13]
        u: torch float32 tensor with shape [batch_size, 4]
        dt: float
        
    returns:
        xn: torch float32 tensor with shape [batch_size, 13]
    """
    # YOUR CODE HERE
    fx = f(x)
    gx = g(x)
    gu = torch.bmm(gx, u.unsqueeze(-1)).squeeze(-1)
    xdot = fx + gu
    xn = x + xdot * dt
    return xn

def roll_out(x0, u_fn, nt, dt):
    """
    Return the state trajectories xts obtained by rolling out the system
    with nt discrete Euler steps using a time step of dt starting at
    states x0 and applying the controller u_fn.
    Note: The returned state trajectories should start with x1; i.e., omit x0.
    Hint: You should use the previous function, euler_step(x, u, dt).

    args:
        x0: torch float32 tensor with shape [batch_size, 13]
        u_fn: Callable u=u_fn(x)
            u_fn takes a torch float32 tensor with shape [batch_size, 13]
            and outputs a torch float32 tensor with shape [batch_size, 4]
        nt: int
        dt: float

    returns:
        xts: torch float32 tensor with shape [batch_size, nt, 13]
    """
    # YOUR CODE HERE
    x = x0
    xs = []

    for _ in range(nt):
        u = u_fn(x)
        x = euler_step(x, u, dt)
        xs.append(x)
    
    xts = torch.stack(xs, dim=1)

    return xts


import cvxpy as cp
from part1 import control_limits


def u_qp(x, h, dhdx, u_ref, gamma, lmbda):
    """
    Return the solution of the CBF-QP with parameters gamma and lmbda
    for the states x, CBF values h, CBF gradients dhdx, and reference controls u_nom.
    Hint: consider using CVXPY to solve the optimization problem: https://www.cvxpy.org/version/1.2/index.html
        Note: We are using an older version of CVXPY (1.2.1) to use the neural CBF library.
            Make sure you are looking at the correct version of documentation.
        Note: You may want to use control_limits() from Part 1.
    Hint: If you use multiple libraries, make sure to properly handle data-type conversions.
        For example, to safely convert a torch tensor to a numpy array: x = x.detach().cpu().numpy()

    args:
        x: torch float32 tensor with shape [batch_size, 13]
        h: torch float32 tensor with shape [batch_size]
        dhdx: torch float32 tensor with shape [batch_size, 13]
        u_ref: torch float32 tensor with shape [batch_size, 4]
        gamma: float
        lmbda: float

    returns:
        u_qp: torch float32 tensor with shape [batch_size, 4]
    """
    umax, umin = control_limits()

    u_min = umin.detach().cpu().numpy().reshape(-1)
    u_max = umax.detach().cpu().numpy().reshape(-1)

    batch_size = x.shape[0]
    u_out = []

    for i in range(batch_size):
        x_i = x[i:i+1]
        h_i = h[i]
        dhdx_i = dhdx[i]
        u_ref_i = u_ref[i].detach().cpu().numpy().reshape(-1)

        f_i = f(x_i).squeeze(0)
        g_i = g(x_i).squeeze(0)

        a = torch.matmul(dhdx_i, g_i).detach().cpu().numpy().reshape(-1)
        b = (torch.dot(dhdx_i, f_i) + gamma * h_i).item()

        u_var = cp.Variable(4)
        delta = cp.Variable(nonneg=True)

        objective = cp.Minimize(cp.sum_squares(u_var - u_ref_i) + lmbda * cp.square(delta))

        # print("u_min shape:", u_min.shape, u_min)
        # print("u_max shape:", u_max.shape, u_max)
        # print("a shape:", a.shape, a)
        # print("u_ref_i shape:", u_ref_i.shape, u_ref_i)

        constraints = [a @ u_var + b + delta >= 0,
            u_var >= u_min, 
            u_var <= u_max]

        prob = cp.Problem(objective, constraints)
        prob.solve()
        # print(prob.status)
        # print(u_var.value)

        u_sol = u_var.value
        u_out.append(torch.tensor(u_sol, dtype=torch.float32))

    return torch.stack(u_out, dim=0)