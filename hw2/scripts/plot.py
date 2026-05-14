import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from plot_utils import failure_mask, roll_out

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
USE_SOLUTIONS = False
if USE_SOLUTIONS:
    from solutions.problem4 import optimal_control
else:
    from problem4 import optimal_control
from problem4_helper import NeuralVF
neuralvf = NeuralVF(ckpt_path='outputs/vf.ckpt')

fig, ax = plt.subplots()
# ax.set_title('$V(x)$ for x=(., ., 0, 0, 0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0)')
ax.set_title(r'$V(x)$ for $x=(p_{x1}, p_{y1}, 0, 8, 0, 0, 0, 8)$')
ax.set_xlabel('$p_x$ (m)')
ax.set_ylabel('$p_y$ (m)')
px = torch.linspace(-3, 3, 100)
py = torch.linspace(-3, 3, 100)
# slice = torch.tensor([
#     0., 0., 0.,
#     1., 0., 0., 0.,
#     5., 0., 0.,
#     0., 0., 0.
# ])
state_slice = torch.tensor([
    0., 0., 0., 8.,   # car 1: px1, py1, psi1, v1
    0., 0., 0., 8.,   # car 2: px2, py2, psi2, v2
])

print('creating plot...')
# values
# PX, PY = torch.meshgrid(px, py, indexing='ij')
# X = torch.zeros((len(px), len(py), 13))
# X[..., 0] = PX
# X[..., 1] = PY
# X[..., 2:] = slice[2:]
px = torch.linspace(-10, 10, 100)
py = torch.linspace(-10, 10, 100)

PX, PY = torch.meshgrid(px, py, indexing='ij')

X = torch.zeros(len(px), len(py), 8)

X[..., 0] = PX      # px1
X[..., 1] = PY      # py1
X[..., 2] = 0.0     # psi1
X[..., 3] = 8.0     # v1

X[..., 4] = 0.0     # px2
X[..., 5] = 0.0     # py2
X[..., 6] = 0.0     # psi2
X[..., 7] = 8.0     # v2
# V = neuralvf.values(X.reshape(-1, 13)).reshape(len(px), len(py))
#For twoCar8D
V = neuralvf.values(X.reshape(-1, 8)).reshape(len(px), len(py))

px = px.detach().cpu().numpy()
py = py.detach().cpu().numpy()
V = V.detach().cpu().numpy()
vbar=3
im = ax.pcolormesh(px, py, V.T, cmap='RdBu', vmin=-vbar, vmax=vbar)
fig.colorbar(im)
ax.contour(px, py, V.T, colors='k', levels=[0])
ax.contour(px, py, X[..., :2].norm(dim=-1)-0.5, colors='r', levels=[0])
# trajectories
# state_min, state_max = torch.clone(slice), torch.clone(slice)
# state_min[0], state_min[1] = -5, -1
# state_max[0], state_max[1] = -2, 1
# x0 = torch.rand(100, 13)*(state_max-state_min)+state_min
# is_safe = neuralvf.values(x0) > 0
# nt = 100
# dt = 0.01
# u_fn = lambda x: optimal_control(x, neuralvf.gradients(x).detach())
# xts = roll_out(x0, u_fn, nt, dt)
# for i, xt in enumerate(xts):
#     ax.plot(xt[:, 0], xt[:, 1], color='green' if is_safe[i] else 'orange')
# safe_line = mlines.Line2D([], [], color='green', label='marked safe')
# unsafe_line = mlines.Line2D([], [], color='orange', label='marked unsafe')
# ax.legend(handles=[safe_line, unsafe_line])
# plt.savefig('outputs/plot.png')
# plt.close()
# print('PLOT SAVED TO outputs/plot.png')

# is_fail = torch.any(failure_mask(xts.reshape(-1, 13)).reshape(len(x0), nt), dim=1)
# false_safety_rate = (torch.sum(is_fail[is_safe])/torch.sum(is_safe)).item()
# print(f'false safety rate: {false_safety_rate}')

#trajectory for twoCar8D
# trajectories
state_min, state_max = torch.clone(state_slice), torch.clone(state_slice)

# sample car 1 initial positions
# state_min[0], state_min[1] = -5, -1
# state_max[0], state_max[1] = -2, 1

# state_min[0], state_min[1] = -5.0, 7.5
# state_max[0], state_max[1] = -2.0, 9.5


# x0 = torch.rand(100, 8) * (state_max - state_min) + state_min
# is_safe = neuralvf.values(x0) > 0

############ Randomized initial positions for both cars
state_min, state_max = torch.clone(state_slice), torch.clone(state_slice)

# car 1 starts outside BRT
state_min[0], state_max[0] = -5.0, -2.0      # px1
state_min[1], state_max[1] = 7.5, 9.5        # py1

# randomize heading and speed of car 1
state_min[2], state_max[2] = -0.8, 0.8       # psi1
state_min[3], state_max[3] = 4.0, 12.0       # v1

# optionally randomize car 2 too
state_min[4], state_max[4] = -1.0, 1.0       # px2
state_min[5], state_max[5] = -1.0, 1.0       # py2
state_min[6], state_max[6] = -0.5, 0.5       # psi2
state_min[7], state_max[7] = 4.0, 12.0       # v2

x0 = torch.rand(100, 8) * (state_max - state_min) + state_min
is_safe = neuralvf.values(x0) > 0



nt = 100
dt = 0.01

# u_fn = lambda x: optimal_control(x, neuralvf.gradients(x).detach())

# xts = roll_out(x0, u_fn, nt, dt)

def roll_out_twocar(x0, nt, dt):
    xts = torch.zeros(x0.shape[0], nt, x0.shape[1])
    xp = x0.clone()

    for i in range(nt):
        xts[:, i] = xp

        dvds = neuralvf.gradients(xp).detach()
        u = neuralvf.dynamics.optimal_control(xp, dvds)
        d = neuralvf.dynamics.optimal_disturbance(xp, dvds)

        xp = xp + dt * neuralvf.dynamics.dsdt(xp, u, d)
        xp = neuralvf.dynamics.equivalent_wrapped_state(xp)

    return xts

xts = roll_out_twocar(x0, nt, dt)

for i, xt in enumerate(xts):
    ax.plot(xt[:, 0], xt[:, 1], color='green' if is_safe[i] else 'orange')

safe_line = mlines.Line2D([], [], color='green', label='marked safe')
unsafe_line = mlines.Line2D([], [], color='orange', label='marked unsafe')
ax.legend(handles=[safe_line, unsafe_line])
plt.savefig('outputs/plot.png')
plt.close()
print('PLOT SAVED TO outputs/plot.png')

is_fail = torch.any(failure_mask(xts.reshape(-1, 8)).reshape(len(x0), nt), dim=1)
false_safety_rate = (torch.sum(is_fail[is_safe])/torch.sum(is_safe)).item()
print(f'false safety rate: {false_safety_rate}')