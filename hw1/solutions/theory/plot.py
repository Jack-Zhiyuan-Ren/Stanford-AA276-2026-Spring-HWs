import numpy as np
import matplotlib.pyplot as plt

# x: [..., 2]
def h(x):
    return 1-np.power(x[..., 0], 2)-np.power(x[..., 1], 2)

def f(x):
    return np.abs(x[..., 0]) - 1.5

z_min, z_max, z_res = -2, 2, 100
v_min, v_max, v_res = -4, 4, 100

zs = np.linspace(z_min, z_max, z_res)
vs = np.linspace(v_min, v_max, v_res)

Zs, Vs = np.meshgrid(zs, vs, indexing='ij')
Xs = np.stack((Zs, Vs), axis=-1)

hs = h(Xs)
fs = f(Xs)

fig, ax = plt.subplots()
im = ax.pcolormesh(zs, vs, hs.T)
fig.colorbar(im)
ax.contour(zs, vs, hs.T, colors='k', levels=[0])
ax.contour(zs, vs, fs.T, colors='r', levels=[0])
ax.set_xlabel('z')
ax.set_ylabel('v')

plt.savefig('solutions/theory/plot.png')