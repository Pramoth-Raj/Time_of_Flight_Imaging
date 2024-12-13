import numpy as np
from . import Estimators as es
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import Slider

def hist_analyser(data):
    """
    Lets you analyse every histogram individually by using sliders
    """
    x_max = data.shape[1]-1
    y_max = data.shape[0]-1
    z_max = data.shape[2]-1
    i, j, k = np.unravel_index(np.argmax(data), data.shape)
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    t = np.linspace(0, z_max, z_max+1)
    s = data[i,j,:]
    l, = ax.plot(t, s, lw=2)

    ax_slider1 = plt.axes(arg=[0.1, 0.1, 0.8, 0.03], facecolor='red')
    ax_slider2 = plt.axes(arg=[0.1, 0.05, 0.8, 0.03], facecolor='blue')
    slider1 = Slider(ax_slider1, label='X_coor', valmin=0, valmax=x_max, valstep=1, valinit=i)
    slider2 = Slider(ax_slider2, label='Y_coor', valmin=0, valmax=y_max, valstep=1, valinit=j)

    def update(value):
        i = slider1.val
        j = slider2.val
        l.set_ydata(data[j,i,:])
        fig.canvas.draw_idle()

    slider1.on_changed(update)
    slider2.on_changed(update)

    plt.show()

def plot_2D(data, bin_width, dimensions=(6, 4), estimator=es.gaus_mle):

    c = 29979245800
    unit_dist = c*bin_width
    h, w, nbins = data.shape
    depth_map = np.zeros((h, w))
    for i in range(h): # create a depth map by using the mean of the time differences
        for j in range(w):
            m = 0
            t = 0
            point_par = estimator(data[i,j])
            depth_map[i,j] = point_par[0]*unit_dist

    x_dim, y_dim = dimensions
    Z = depth_map
    x = np.linspace(0, x_dim, Z.shape[1])  # Generate X values based on Z shape
    y = np.linspace(0, y_dim, Z.shape[0])  # Generate Y values based on Z shape
    X, Y = np.meshgrid(x, y)  # Create the grid

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # Add labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

    return depth_map

def plot_1D(data, bin_width, dimension=6, estimator=es.gaus_mle):

    c = 29979245800
    unit_dist = c*bin_width
    time_delay_1d = np.array(data)
    depth_map = np.zeros(len(time_delay_1d))
    for j in range(len(depth_map)):
            m = 0
            t = 0
            point_par = estimator(time_delay_1d[j])
            depth_map[j] = point_par[0]*unit_dist

    x = np.linspace(0, dimension, len(time_delay_1d))
    plt.figure()
    plt.plot(x, depth_map)
    plt.xlabel('x in cm')
    plt.ylabel('depth in cm')
    plt.title('1D depth plot')
    plt.show()
    
    return depth_map
