import numpy as np
from skimage.io import imread
from skimage.filters import sobel_h, sobel_v, sobel
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from accumulator import DiamondSpace
from matplotlib.lines import Line2D
##
image = imread("c://Users//jurankovam//PycharmProjects//DiamondSpacePackage//chess.png", as_gray=True)
_,ax = plt.subplots(1, figsize=(5,5))
ax.imshow(image, cmap="gray")
ax.set(title="Input image", xticks=[], yticks=[])
plt.tight_layout()
##
v_edges = gaussian_filter(image, [1.5,1.5],[0,1])
h_edges = gaussian_filter(image, [1.5,1.5],[1,0])
gradient = np.arctan2(h_edges, v_edges)

edges = np.sqrt(h_edges * h_edges + v_edges*v_edges)

_,ax = plt.subplots(1,2, figsize=(10,5))
ax[0].imshow(edges, cmap="Greys")
ax[0].set(title="Edge map - observations", xticks=[], yticks=[])
ax[1].imshow(gradient, cmap="twilight_shifted")
ax[1].set(title="Edge map - gradients", xticks=[], yticks=[])
plt.tight_layout()
##
h,w = image.shape[:2]
h_half = np.median(np.arange(h))
w_half = np.median(np.arange(w))
coord_x, coord_y = np.meshgrid(np.arange(w) - w_half, np.arange(h) - h_half)
cos_g = np.cos(gradient)
sin_g = np.sin(gradient)

idx = edges > 0.2

A = cos_g[idx]
B = sin_g[idx]
C = -sin_g[idx] * coord_y[idx] - cos_g[idx] * coord_x[idx]
lines = np.transpose(np.vstack((A, B, C)))

##
space_size = 128
d = 256
DS = DiamondSpace(d, space_size)
DS.insert(lines)
##
A = DS.attach_spaces()
_,ax = plt.subplots(1, figsize=(5,5))
ax.imshow(A, cmap="inferno", extent = (-DS.size,DS.size,-DS.size,DS.size))
ax.set(title="Accumulator", xticks=np.linspace(-DS.size,DS.size,5), yticks=np.linspace(-DS.size,DS.size,5))
plt.tight_layout()
ax.axis('on')
##
_,ax = plt.subplots(2,2, figsize=(5,5))

Amax = A.max()
ax[0,0].axis('off')
ax[0,1].axis('off')
ax[1,1].axis('off')
ax[1,0].axis('off')

ax[0,0].imshow(DS.A[0], cmap="inferno", vmin = 0, vmax = Amax)
ax[0,0].invert_yaxis()
ax[0,0].invert_xaxis()
ax[0,1].imshow(DS.A[1], cmap="inferno", vmin = 0, vmax = Amax)
ax[0,1].invert_yaxis()
ax[1,1].imshow(DS.A[2], cmap="inferno", vmin = 0, vmax = Amax)
ax[1,0].imshow(DS.A[3], cmap="inferno", vmin = 0, vmax = Amax)
ax[1,0].invert_xaxis()

ax[0,0].text(20, 10, "ST", fontsize=10, color = "white")
ax[0,1].text(10, 10, "SS", fontsize=10, color = "white")
ax[1,1].text(10, 16, "TS", fontsize=10, color = "white")
ax[1,0].text(20, 16, "TT", fontsize=10, color = "white")
##

##
p, w, p_ds = DS.find_peaks(min_dist=2, prominence=2, t=0.8)
sorted = np.argsort(w)
p = p[sorted[::-1],:]
w = w[sorted[::-1]]
p_ds = p_ds[sorted[::-1],:]

##
A = DS.attach_spaces()
_,ax = plt.subplots(1, figsize=(5,5))
ax.imshow(A, cmap="inferno", extent = (-DS.size+0.5,DS.size-0.5,DS.size-0.5,-DS.size+0.5))
ax.set(title="Accumulator", xticks=np.linspace(-DS.size+1,DS.size-1,5), yticks=np.linspace(-DS.size+1,DS.size-1,5))

ax.plot(p_ds[:,0], p_ds[:,1], "r+")

ax.invert_yaxis()
plt.tight_layout()
ax.axis('on')

## dorobit scale od -d po d
p, w, p_ds = DS.find_peaks(min_dist=2, prominence=2, t=0.8)
sorted = np.argsort(w)
p = p[sorted[::-1],:]
w = w[sorted[::-1]]
p_ds = p_ds[sorted[::-1],:]
scale = (DS.size - 1) / DS.d


A = DS.attach_spaces()
_,ax = plt.subplots(1, figsize=(5,5))
ax.imshow(A, cmap="inferno", extent = (-d+0.5/scale,d-0.5/scale,d-0.5/scale,-d+0.5/scale))
ax.set(title="Accumulator", xticks=np.linspace(-d+1,d-1,5), yticks=np.linspace(-d+1,d-1,5))
ax.plot(p_ds[:,0]/scale, p_ds[:,1]/scale, "r+")

ax.invert_yaxis()
plt.tight_layout()
ax.axis('on')