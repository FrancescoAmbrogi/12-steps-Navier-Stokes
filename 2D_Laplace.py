### This program aim at resolving the 2D Laplace equation
### Created by Francesco Ambrogi on March the 5th

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

### Let's create a function
def plot2D(x,y,p):
    fig = plt.figure(figsize=(11,7), dpi=100)
    ax = fig.gca(projection='3d')
    X,Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,p[:], rstride=1,cstride=1,cmap=cm.coolwarm,linewifth=0,antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1)
    ax.view_init(30,225)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


    