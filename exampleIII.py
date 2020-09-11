import numpy as np
import accumulator as ac
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

## generate N random points
N = 10000
rad = 100000000
P = np.random.rand(N,3)*2*rad - rad
P[:,2] = 1
##
def get_colors(p):

    min_x = p[:, 0].min()
    max_x = p[:, 0].max()
    min_y = p[:, 1].min()
    max_y = p[:, 1].max()

    colors = np.ones([p.shape[0],3])

    colors[:,0] = (p[:, 0] - min_x) / (max_x - min_x)
    colors[:,1] = (p[:,1] - min_y)/(max_y - min_y)
    colors[:,2] = 0


    return colors

##
d = 20000000
size = 128
DS = ac.DiamondSpace(d, size)

_,ax = plt.subplots(1,2, figsize=(10,5))

co = get_colors(P)
ax[0].scatter(P[:,0],P[:,1], c = co)
ax[0].set(title="Cartesian CS", xticks=np.linspace(-rad,rad,5), yticks=np.linspace(-rad,rad,5))

P_I = DS.points_to_ds(P)
ax[1].scatter(P_I[:,0],P_I[:,1], c = co)
ax[1].set(title="Cartesian CS", xticks=np.linspace(-d,d,5), yticks=np.linspace(-d,d,5))


#draw_lines(lines,W, ax[1])
#draw_lines(lines,np.ones(lines.shape[0]), ax[1])
#ax[1].plot(peaks[:,0],peaks[:,1] ,"*",c = "red")

