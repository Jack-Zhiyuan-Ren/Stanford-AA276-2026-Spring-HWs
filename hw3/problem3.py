import torch
import cvxpy as cp
from problem3_helper import control_limits, f, g

from problem3_helper import NeuralVF
vf = NeuralVF()

# environment setup
obstacles = torch.tensor([
    [1.0,  0.0, 0.5], # [px, py, radius]
    [4.0,  2.0, 1.0],
    [4.0, -2.0, 1.0],
    [7.0,  0.0, 1.5],
    [7.0,  4.0, 0.5],
    [7.0, -4.0, 0.5]
])

def smooth_blending_safety_filter(x, u_nom, gamma, lmbda):
    """
    Compute the smooth blending safety filter.
    Refer to the definition provided in the handout.
    You might find it useful to use functions from
    previous homeworks, which we have imported for you.
    These include:
      control_limits(.)
      f(.)
      g(.)
      vf.values(.)
      vf.gradients(.)
    NOTE: some of these functions expect batched inputs,
    but x, u_nom are not batched inputs in this case.
    
    args:
        x:      torch tensor with shape [13]
        u_nom:  torch tensor with shape [4]
        
    returns:
        u_sb:   torch tensor with shape [4]
    """
    # YOUR CODE HERE
    x_cpu = x.detach().float().cpu()
    u_nom_cpu = u_nom.detach().float().cpu()

    #Adapt the original value function with the new obstacles
    num_obs = obstacles.shape[0]
    x_query = x_cpu.unsqueeze(0).repeat(num_obs, 1) #One state for each obstacle


    # Shifting the original obstacle to each of the new obstacle
    x_query[:, 0] = x_query[:, 0] - obstacles[:, 0]  # px - ox
    x_query[:, 1] = x_query[:, 1] - obstacles[:, 1]  # py - oy

    values = vf.values(x_query)
    # account for the new radiuses
    values = values - (obstacles[:, 2] - 0.5)

    #looking for minimum values across all obstacles
    active_idx = torch.argmin(values)
    V = values[active_idx].item()

    gradients = vf.gradients(x_query)
    dVdx = gradients[active_idx]  # shape [13]

    #f(x) and g(x)
    x_batch = x_cpu.unsqueeze(0)

    fx = f(x_batch)[0]      # shape [13]
    gx = g(x_batch)[0]      # shape [13, 4]

    dVdx_np = dVdx.numpy()
    fx_np = fx.numpy()
    gx_np = gx.numpy()
    u_nom_np = u_nom_cpu.numpy()

    A = dVdx_np @ gx_np     # shape [4]
    b = dVdx_np @ fx_np     # scalar

    # Solving the QP with CVXPY
    upper, lower = control_limits()
    upper_np = upper.numpy()
    lower_np = lower.numpy()

    u = cp.Variable(4)
    s = cp.Variable(nonneg=True)

    objective = cp.Minimize(
        cp.sum_squares(u - u_nom_np) + lmbda * cp.square(s)
    )

    constraints = [
        A @ u + b + gamma * V + s >= 0,
        u >= lower_np,
        u <= upper_np,
    ]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.OSQP, warm_start=True)
    except Exception:
        prob.solve(warm_start=True)

    # Fallback in case solver fails for numerical reasons
    if u.value is None:
        u_fallback = torch.clamp(u_nom_cpu, lower, upper)
        return u_fallback.to(dtype=torch.float32)


    
    

    return torch.tensor(u.value, dtype=torch.float32) # NOTE: ensure you return a float32 tensor