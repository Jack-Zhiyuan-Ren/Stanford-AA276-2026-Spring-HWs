import numpy as np
import jax.numpy as jnp

import hj_reachability as hj
from hj_reachability import dynamics
from hj_reachability import sets

from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for saving figures

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from problem3_helper import save_values_gif, plot_value_and_safe_set_boundary

class InvertedPendulum(dynamics.ControlAndDisturbanceAffineDynamics):

  def __init__(self,
               m=2.,
               l=1.,
               g=10.,
               u_bar=3.):
    self.m = m
    self.l = l
    self.g = g
    control_mode = 'max'
    disturbance_mode = 'min'
    control_space = sets.Box(jnp.array([-u_bar]), jnp.array([u_bar]))
    disturbance_space = sets.Box(jnp.array([0.]), jnp.array([0.]))
    super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

  def open_loop_dynamics(self, state, time):
    theta, theta_dot = state
    return jnp.array([
      theta_dot,
      self.g * jnp.sin(theta) / self.l
    ])

  def control_jacobian(self, state, time):
    return jnp.array([[0.], [1. / (self.m*(self.l**2))]])

  def disturbance_jacobian(self, state, time):
    return jnp.array([[0.], [0.]])
  

###### PROBLEM 3.1 AND 3.2
inverted_pendulum_dynamics = InvertedPendulum()
grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
  hj.sets.Box(np.array([-np.pi, -10.]),
              np.array([+np.pi, +10.])),
  (101, 101)
)
failure_values = 0.3 - jnp.abs(grid.states[..., 0])
times = np.linspace(0, -5, 101, endpoint=True)
solver_settings = hj.SolverSettings.with_accuracy('very_high',
  hamiltonian_postprocessor=hj.solver.backwards_reachable_tube
)
values = hj.solve(solver_settings, inverted_pendulum_dynamics, grid, times, failure_values)
save_values_gif(values, grid, times, save_path='solutions/outputs/values.gif')


###### PROBLEM 3.3
values_converged = values[-1]
values_converged_interpolator = RegularGridInterpolator(
  ([np.array(v) for v in grid.coordinate_vectors]),
  np.array(values_converged),
  bounds_error=False,
  fill_value=None
)
num_samples = int(1e7)
batch_size = int(1e5)
num_batches = int(num_samples/batch_size)
assert num_batches*batch_size == num_samples
sample_min = np.array([-np.pi, -10])
sample_max = np.array([+np.pi, +10])
num_safe = 0
for _ in tqdm(range(num_batches)):
  samples = np.random.uniform(
    low=sample_min,
    high=sample_max,
    size=(batch_size, len(sample_min))
  )
  num_safe += np.sum(values_converged_interpolator(samples) > 0)
print(f'Safe Set Volume: {num_safe*np.prod(sample_max-sample_min)/num_samples}')


###### PROBLEM 3.4
grads_converged = grid.grad_values(values_converged, solver_settings.upwind_scheme)
beta2s_converged = grads_converged[:, :, 1]
beta2s_converged_interpolator = RegularGridInterpolator(
  ([np.array(v) for v in grid.coordinate_vectors]),
  np.array(beta2s_converged),
  bounds_error=False,
  fill_value=None
)
def euler_step(x, u, dt=0.01):
  return x + dt*np.array([x[1].item(), (10*np.sin(x[0])+u/2).item()])
def optimal_safety_controller(x):
  return np.sign(beta2s_converged_interpolator(x))*3
def simulate(x0, nt, dt=0.01):
  xs = np.full((nt+1, 2), fill_value=np.nan)
  us = np.full(nt, fill_value=np.nan)
  xs[0] = x0
  for i in tqdm(range(nt)):
    x = xs[i]
    u = optimal_safety_controller(x)
    xs[i+1] = euler_step(x, u, dt)
    us[i] = u
  return xs, us
x0s = [
  np.array([-0.1, +0.4]),
  np.array([-0.1, -0.3])
]
fig, axes = plt.subplots(2, 1)
plot_value_and_safe_set_boundary(values_converged, grid, axes[0])
axes[0].set_title('State Trajectory')
axes[0].set_xlabel('$\\theta$ (rad)')
axes[0].set_ylabel('$\\dot{\\theta}$ (rad/s)')
axes[1].set_title('Control Profile')
axes[1].set_xlabel('$t$ (s)')
axes[1].set_ylabel('$u$')
dt = 0.01
for x0 in x0s:
  xs, us = simulate(x0, int(1/dt), dt)
  axes[0].plot(xs[:, 0], xs[:, 1], label=f'{x0}')
  axes[1].plot(np.arange(len(us)), us, label=f'{x0}')
axes[0].legend()
axes[1].legend()
fig.tight_layout()
fig.savefig('solutions/outputs/simulation.png')