import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import least_squares
from scipy.interpolate import interp1d

def read_polylines_from_csv(file_path):
    polylines = []
    with open (file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            polyline = [(float(row[i]), float(row[i+1])) for i in range(0, len(row), 2)]
            polylines.append(np.array(polyline))

    return polylines

def plot_polylines(polylines):
    for polyline in polylines:
        plt.plot(polyline[:, 0], polyline[:, 1], marker='o')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


polylines = read_polylines_from_csv('frag0.csv')
plot_polylines(polylines)


#Regularizing curve 

def detect_line(polyline):
    x = polyline[:, 0].reshape(-1, 1)
    y = polyline[:, 1]
    reg = LinearRegression().fit(x,y)
    y_pred = reg.predict(x)
    

    plt.plot(polyline[:, 0], polyline[:, 1], marker='o', label = 'Original')
    plt.plot(polyline[:,0], y_pred, label ='Fitted Line')

    plt.gca().set_aspect('equal', adjustable ='box')
    plt.legend()
    plt.show()


detect_line(polylines[0])

def ellipse_residuals(params, x, y):
    xc, yc, a, b, theta = params
    cost = ((x - xc) * np.cos(theta) + (y - yc) * np.sin(theta))**2 / a**2 + \
           ((x - xc) * np.sin(theta) - (y - yc) * np.cos(theta))**2 / b**2 - 1
    return cost

def detect_ellipse(polyline):
    x = polyline[:, 0]
    y = polyline[:, 1]
    x_m = np.mean(x)
    y_m = np.mean(y)
    initial_guess = [x_m, y_m, np.std(x), np.std(y), 0]
    result = least_squares(ellipse_residuals, initial_guess, args=(x, y))
    xc, yc, a, b, theta = result.x
    
    ellipse = Ellipse((xc, yc), 2*a, 2*b, np.degrees(theta), edgecolor='r', fc='None', lw=2)
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    ax.add_patch(ellipse)
    ax.plot(x, y, 'o')
    plt.show()

from matplotlib.patches import Ellipse
detect_ellipse(polylines[0])


def detect_symmetry(polyline):
    x = polyline[:, 0]
    y = polyline[:, 1]
    
    xc = np.mean(x)
    yc = np.mean(y)
    
    plt.plot(x, y, 'o')
    plt.axvline(x=xc, color='r', linestyle='--', label='Vertical Symmetry Line')
    plt.axhline(y=yc, color='b', linestyle='--', label='Horizontal Symmetry Line')
    
    plt.legend()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

detect_symmetry(polylines[0])

def complete_curve(polyline, num_points=100):
    x = polyline[:, 0]
    y = polyline[:, 1]
    
    t = np.linspace(0, 1, len(x))
    t_new = np.linspace(0, 1, num_points)
    
    f_x = interp1d(t, x, kind='cubic')
    f_y = interp1d(t, y, kind='cubic')
    
    x_new = f_x(t_new)
    y_new = f_y(t_new)
    
    plt.plot(x, y, 'o', label='Original')
    plt.plot(x_new, y_new, '-', label='Completed Curve')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()

complete_curve(polylines[0])


