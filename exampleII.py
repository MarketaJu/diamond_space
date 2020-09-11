# -*- coding: utf-8 -*-
"""DiamondSpaceExampleII.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ms7aHDozJEok2ytWuPD63i_hdUrT5KG_
"""

import numpy as np
import accumulator as ac
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

##
# generate N random lines around random point P
N = 100
P = np.random.randint(0,2000000000,3) - 100
W = np.random.random(N)
P[2] = 1
X = np.tile(P,(N,1))
Y = np.random.rand(N,3)*100 - 50
Y[:,2] = 0
Y = Y + X
lines = np.cross(X,Y)
print("Random point:",P)


##
# function for plotting generated lines
def draw_lines(lines, w, ax):

  num = lines.shape[0]

  x0 = np.array([1,0,100])
  x1 = np.array([1,0,-100])
  y0 = np.array([0,1,100])
  y1 = np.array([0,1,-100])

  p_x0 = np.cross( np.tile(x0,(num,1)),lines)
  p_x1 = np.cross( np.tile(x1,(num,1)),lines)
  p_y0 = np.cross( np.tile(y0,(num,1)),lines)
  p_y1 = np.cross( np.tile(y1,(num,1)),lines)

  p_x0 = np.divide(p_x0, p_x0[:, 2:], where=~np.isclose(p_x0[:, 2:], 0))
  p_x1 = np.divide(p_x1, p_x1[:, 2:], where=~np.isclose(p_x1[:, 2:], 0))
  p_y0 = np.divide(p_y0, p_y0[:, 2:], where=~np.isclose(p_y0[:, 2:], 0))
  p_y1 = np.divide(p_y1, p_y1[:, 2:], where=~np.isclose(p_y1[:, 2:], 0))

  theta = np.arctan2(lines[:,1],lines[:,0])
  horizontal = np.abs(np.sin(theta)) >= np.sqrt(2)/2

  lin_seg = np.ndarray([num, 2, 2])
  lin_seg[horizontal, 0, 0:2] = p_x0[horizontal, 0:2]
  lin_seg[horizontal, 1, 0:2] = p_x1[horizontal, 0:2]
  lin_seg[~horizontal, 0, 0:2] = p_y0[~horizontal, 0:2]
  lin_seg[~horizontal, 1, 0:2] = p_y1[~horizontal, 0:2]

  colors = np.ones([num,4])*0.5
  colors[:,3] = w

  lc = mc.LineCollection(lin_seg, colors=colors, linewidths=1, zorder=1)
  ax.add_collection(lc)

  ax.set(xlim=(-100, 100), ylim=(-100, 100))
##
# create Diamond space, accumulate lines and find peaks
d = 2000000000
size = 128
DS = ac.DiamondSpace(d, size)
DS.insert(lines)

peaks, weights, peaks_ds = DS.find_peaks(min_dist=2, prominence=2, t=0.8)

# draw accumulator with detected max peaks (peaks_ds) and original image with reprojected peaks (peaks)
_,ax = plt.subplots(1,2, figsize=(10,5))

ax[0].imshow(np.sqrt(DS.attach_spaces()), cmap="Greys", extent = (-DS.size+0.5,DS.size-0.5,DS.size-0.5,-DS.size+0.5))
ax[0].set(title="Accumulator", xticks=np.linspace(-DS.size,DS.size,5), yticks=np.linspace(-DS.size,DS.size,5))
ax[0].plot(peaks_ds[:,0],peaks_ds[:,1],"*",c = "red")
ax[0].invert_yaxis()

#draw_lines(lines,W, ax[1])
draw_lines(lines,np.ones(lines.shape[0]), ax[1])
ax[1].plot(peaks[:,0],peaks[:,1] ,"*",c = "red")

print("Generated point:",P)
print("Detected point:",peaks)

##
# create Diamond space, accumulate lines and find peaks
d = 100
size = 128
DS = ac.DiamondSpace(d, size)
DS.insert(lines)
DSW = ac.DiamondSpace(d, size)
DSW.insert(lines,W)

##
##
_,ax = plt.subplots(2,2, figsize=(5,5))

A = DS.attach_spaces()
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
# just visualization of the separated parts of the accumulator

_,ax = plt.subplots(2,2, figsize=(5,5))

Amax =1
ax[0,0].axis('off')
ax[0,1].axis('off')
ax[1,1].axis('off')
ax[1,0].axis('off')

ax[0,0].imshow(DS.A[0], cmap="cool", vmin = 0, vmax = Amax)
ax[0,0].invert_yaxis()
ax[0,0].invert_xaxis()
ax[0,1].imshow(DS.A[1], cmap="cool", vmin = 0, vmax = Amax)
ax[0,1].invert_yaxis()
ax[1,1].imshow(DS.A[2], cmap="cool", vmin = 0, vmax = Amax)
ax[1,0].imshow(DS.A[3], cmap="cool", vmin = 0, vmax = Amax)
ax[1,0].invert_xaxis()

ax[0,0].text(20, 10, "ST", fontsize=10, color = "black")
ax[0,1].text(10, 10, "SS", fontsize=10, color = "black")
ax[1,1].text(10, 16, "TS", fontsize=10, color = "black")
ax[1,0].text(20, 16, "TT", fontsize=10, color = "black")

##

A = DS.attach_spaces()
_,ax = plt.subplots(1, figsize=(5,5))
ax.imshow(A, cmap="cool", extent = (-DS.size,DS.size,DS.size,-DS.size))
ax.set(title="Accumulator", xticks=np.linspace(-DS.size,DS.size,5), yticks=np.linspace(-DS.size,DS.size,5))
ax.invert_yaxis()
plt.tight_layout()