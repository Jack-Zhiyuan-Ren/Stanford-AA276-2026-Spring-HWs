print('Run ../hw1/solutions/homework2_problem4_4.py to estimate the volume of the safe set for the trained CBF model.')
print('Estimating the volume of the safe set for the trained value function model...')

import torch
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from problem4_helper import NeuralVF

neuralvf = NeuralVF()
state_min = torch.tensor([-3, -3, -3, -1, -1, -1, -1, -5, -5, -5, -5, -5, -5])
state_max = torch.tensor([3, 3, 3, 1, 1, 1, 1, 5, 5, 5, 5, 5, 5])
n_samples = int(1e7)
batch_size = int(1e5)
n_batches = n_samples // batch_size
assert n_batches * batch_size == n_samples, "n_samples must be divisible by batch_size"
num_safe = 0
for i in tqdm(range(n_batches)):
    x = torch.rand(batch_size, 13) * (state_max - state_min) + state_min
    num_safe += (neuralvf.values(x) > 0).sum().item()
print(f"Estimated volume of the safe set: {num_safe / n_samples:.4f}")