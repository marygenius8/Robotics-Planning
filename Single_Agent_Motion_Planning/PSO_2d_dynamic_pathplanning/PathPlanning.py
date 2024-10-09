"""
Path Planning Using Particle Swarm Optimization


Implementation of particle swarm optimization (PSO) for path planning when the
environment is known.


Copyright (c) 2021 Gabriele Gilardi


Main Quantities
---------------
start           Start coordinates.
goal            Goal coordinates.
limits          Lower and upper boundaries of the layout.
obs             List containing the obstacles parameters.
f_interp        Type of spline (slinear, quadratic, cubic).
nPts            Number of internal points defining the spline.
Px, Py          Spline coordinates.
L               Path length.
F               Function to minimize.
err             Penalty term.
count           Number of violated obstacles.
sol             Tuple containing the solution.
ns              Number of points defining the spline.
X               Array of variables.
Xinit           Initial value of the variables.
LB              Lower boundaries of the search space.
UB              Upper boundaries of the search space.
nVar            Number of variables (equal to twice nPts).
nPop            Number of agents (one for each path).
epochs          Number of iterations.
K               Average size of each agent's group of informants.
phi             Coefficient to calculate the two confidence coefficients.
vel_fact        Velocity factor to calculate the maximum and the minimum
                allowed velocities.
conf_type       Confinement type (on the velocities).
IntVar          List of indexes specifying which variable should be treated
                as integers.
normalize       Specifies if the search space should be normalized (to
                improve convergency).
rad             Normalized radius of the hypersphere centered on the best
                particle.
args            List containing the parameters needed for the calculation of
                the function to minimize.
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle, Polygon
from pso import PSO


def build_Xinit(start, goal, nPts):
    """
    Returns the straight path between start and goal position in the correct
    format for array <Xinit>.
    """
    xs, ys = start
    xg, yg = goal

    Px = np.linspace(xs, xg, nPts+2)
    Py = np.linspace(ys, yg, nPts+2)

    Xinit = np.concatenate((Px[1:-1], Py[1:-1]))

    return Xinit


def centroid(V):
    """
    Returns the position of the centroid of a polygon defined by array <V>.
    The vertices are assumed given in counter-clockwise order.

    Reference: http://en.wikipedia.org/wiki/Centroid
    """
    V = np.asarray(V)
    nPts = len(V)

    xc = 0.0        # Centroid x-coordinate
    yc = 0.0        # Centroid y-coordinate
    A = 0.0         # Polygon area

    for i in range(nPts):

        d = V[i-1][0] * V[i][1] - V[i][0] * V[i-1][1]
        xc += (V[i-1][0] + V[i][0]) * d
        yc += (V[i-1][1] + V[i][1]) * d
        A += d

    A = A / 2.0
    xc = xc / (6.0 * A)
    yc = yc / (6.0 * A)

    return xc, yc


def cal_collision_penalty(obs, Px, Py):
    """
    Returns a penalty value if any point of the path violates any of the
    obstacles. To speed up the calculation the algorithms have been designed
    to work on all points simultaneously.

    Notes:
    - Polygon verteces must be given in counter-clockwise order.
    - "Ellipse" can default to a circular obstacle, but "Circle" is faster.
    - "Polygon" can default to a convex polygonal obstacle, but "Convex" is
       faster.
    - Each path is defined by a row in <Px> and <Py>.

    Reference: http://paulbourke.net/geometry/polygonmesh/
    """
    err = np.zeros(Px.shape[0])     # the dimension of Px and Py agents(nPop)*sampled points in planned path
    count = 0

    # Loop over all obstacle
    for name, obs_set in obs.items():
        # Obstacle data
        for data in obs_set:
            # Obstacle type and its centroid
            xc, yc = data[:2]

            # Distances from the obstacle centroid
            d = np.sqrt((Px - xc) ** 2 + (Py - yc) ** 2)

            # Obstacle is a circle (r = radius, Kv = scaling factor)
            if (name == 'Circle'):
                r, Kv = data[2:]
                inside = r*2 > d

            # Obstacle is an ellipse (theta = semi-major axis rotation from the
            # x-axis, b = semi-minor axis, e = eccentricity, Kv = scaling factor).
            elif (name == 'Ellipse'):
                theta, b, e, Kv = data[2:]
                angle = np.arctan2(Py-yc, Px-xc) - theta
                r = b / np.sqrt(1.0 - (e * np.cos(angle)) ** 2)
                inside = r*2 > d

            # Obstacle is a convex polygon (V = vertices, Kv =scaling factor)
            elif (name == 'Convex'):
                V, Kv = data[2:]
                a = np.ones(Px.shape) * np.inf
                for i in range(V.shape[0]):
                    side = (Py - V[i-1, 1]) * (V[i, 0] - V[i-1, 0]) \
                           - (Px - V[i-1, 0]) * (V[i, 1] - V[i-1, 1])
                    a = np.minimum(a, side)
                inside = a > 0.0

            # Obstacle is a polygon (V = vertices, Kv = scaling factor)
            elif (name == 'Polygon'):
                V, Kv = data[2:]
                inside = np.zeros(Px.shape, dtype=bool)
                for i in range(V.shape[0]):
                    a = ((V[i, 1] > Py) != (V[i-1, 1] > Py)) & \
                        (Px < (V[i, 0] + (V[i-1, 0] - V[i, 0]) * (Py - V[i, 1]) /
                                          (V[i-1, 1] - V[i, 1])))
                    inside = np.where(a, np.logical_not(inside), inside)

            # Penalty values
            penalty = np.where(inside, Kv / d, 0.0)

            #  Update the number of obstacles violated
            if (inside.any()):
                count += 1

            # The penalty of each path is taken as the average penalty between its
            # inside and outside points
            err += np.nanmean(penalty, axis=1)

    return err, count


def fitness(X, args):
    """
    Returns the function to minimize, i.e. the path length when there is
    not any obstacle violation.

    The interpolation method can be "slinear", "quadratic", or "cubic" (spline
    of order 1, 2, and 3, respectively). The curvilinear coordinate along the
    path is taken in the interval from 0 (start) to 1 (goal).
    """
    # Arguments passed
    Xs, Ys = args[0]            # Start position (as array)
    Xg, Yg = args[1]            # Goal position (as array)
    obs = args[2]               # List of obstacles
    ns = args[3]                # Number of points along the spline
    f_interp = args[4]          # Interpolation method
    time_step = 0.1
    max_vxyz = 8.0  # [m/s]
    max_vxy = 6.0
    max_vz = 4.0
    min_vz = -3.0
    max_uxyz = 8.0  # [m/s^2]
    max_uxy = 8.0
    max_uz = 4.0

    # smooth_penalty = np.zeros(args[0][0].shape[0])
    # jerk_penalty = np.zeros(args[0][0].shape[0])

    w_l = 1.0   # path length penalty weight
    w_g = 1.0
    w_c = 10.0
    w_s = 0.1
    w_f = 0.1

    nPop, nVar = X.shape
    nPts = nVar // 2            # Number of (internal) breakpoints

    # Coordinates of the breakpoints (start + internal + goal)
    x = np.block([Xs, X[:, :nPts], Xg])
    y = np.block([Ys, X[:, nPts:], Yg])

    # Classes defining the spline
    t = np.linspace(0, 1, nPts+2)
    CSx = interp1d(t, x, axis=1, kind=f_interp, assume_sorted=True)
    CSy = interp1d(t, y, axis=1, kind=f_interp, assume_sorted=True)

    # Coordinates of the discretized path
    s = np.linspace(0, 1, ns)
    Px = CSx(s)
    Py = CSy(s)

    # get states and control trajectories
    dX = np.diff(Px, axis=1)
    dY = np.diff(Py, axis=1)

    Vx = dX / time_step
    Vy = dY / time_step
    Vxy = np.linalg.norm(np.stack((Vx, Vy), axis=-1), axis=-1)

    Ux = np.diff(Vx, axis=1) / time_step
    Uy = np.diff(Vy, axis=1) / time_step
    Uxy = np.linalg.norm(np.stack((Ux, Uy), axis=-1), axis=-1)

    # calculate Path length
    L = np.sqrt(dX ** 2 + dY ** 2).sum(axis=1)

    # calculate Goal penalty value
    current_position = np.transpose(np.array([Px[:,-1], Py[:,-1]]))
    goal = np.hstack((args[1][0], args[1][1]))
    goal_penalty = np.linalg.norm(current_position - goal, axis=1)

    # calculate Collision penalty value
    collision_penalty, count = cal_collision_penalty(obs, Px, Py)

    # Calculate Smooth penalty value
    Ux_jerk = np.diff(Ux, axis=1)
    Uy_jerk = np.diff(Uy, axis=1)
    smooth_penalty = np.sum(Ux_jerk ** 2 +  Uy_jerk ** 2, axis=1)

    # Calculate Jerk penalty value
    jerk_penalty = np.sum(np.diff(Ux_jerk) ** 2 + np.diff(Uy_jerk) ** 2, axis=1)

    # calculate Feasiblity penalty value
    acceleration_penalty = np.sum(np.maximum(0, np.abs(Uxy) - max_uxy) ** 2, axis=1)
    velocity_penalty = np.sum(np.maximum(0, np.abs(Vxy) - max_vxy) ** 2, axis=1)

    # Function to minimize
    fitness = L * (1.0 + collision_penalty) + w_s * (smooth_penalty + jerk_penalty) + w_f * (acceleration_penalty + velocity_penalty)
    # fitness = w_l*L + w_g * goal_penalty + w_c * collision_penalty + w_s * (smooth_penalty) + w_f * (acceleration_penalty + velocity_penalty)

    # Return the results for the best path if it is the last call
    if (len(args) == 6):
        args[5] = [L, count, Px, Py]    # if this args can be returned?

    return fitness    # cost/fitness value/objective function value


class PathPlanning:
    """
    Class path optimization.
    """
    def __init__(self, start=None, goal=None, limits=None):
        """
        Initialize the object.
        """
        self.start = None if (start is None) else np.asarray(start)
        self.goal = None if (goal is None) else np.asarray(goal)
        self.limits = None if (limits is None) else np.asarray(limits)
        self.obs = dict()

    def __repr__(self):
        """
        Returns the string representation of the PathPlanning object.
        """
        return ("\nPathPlanning object \
                 \n- start = {} \
                 \n- goal = {} \
                 \n- limits = {} \
                 \n- number of obstacles = {}" \
                .format(self.start, self.goal, self.limits, len(self.obs)))

    def obs_info(self):
        """
        Prints information about the obstacles.
        """
        nObs = len(self.obs)
        if (nObs > 0):
            print("\n===== Obstacles =====")
        else:
            print("\nNo obstacles defined.")

        # Loop over all obstacle
        for i in range(nObs):

            # Obstacle data
            data = self.obs[i]

            # Obstacle type and its centroid
            name, xc, yc = data[:3]

            # Obstacle is a circle
            if (name == 'Circle'):
                r, Kv = data[3:]
                print("\n{} \
                       \n- centroid = {} \
                       \n- radius = {} \
                       \n- scaling factor = {}" \
                       .format(name, (xc, yc), r, Kv))

            # Obstacle is an ellipse (e = eccentricity)
            elif (name == 'Ellipse'):
                theta, b, e, Kv = data[3:]
                theta = theta * 180.0 / np.pi
                a = b / np.sqrt(1.0 - e ** 2)
                print("\n{} \
                       \n- centroid = {} \
                       \n- rotation from x-axis= {} \
                       \n- semi-major axis = {} \
                       \n- semi-minor axis = {} \
                       \n- scaling factor = {}" \
                       .format(name, (xc, yc), theta, a, b, Kv))

            # Obstacle is a convex polygon
            elif (name == 'Convex'):
                V, Kv = data[3:]
                print("\n{} \
                       \n- centroid = {} \
                       \n- vertices =\n{} \
                       \n- scaling factor = {}" \
                       .format(name, (xc, yc), V.T, Kv))

            # Obstacle is a polygon
            elif (name == 'Polygon'):
                V, Kv = data[3:]
                print("\n{} \
                       \n- centroid = {} \
                       \n- vertices =\n{} \
                       \n- scaling factor = {}" \
                       .format(name, (xc, yc), V.T, Kv))

    def set_start(self, x, y):
        """
        Sets the start position.
        """
        self.start = np.array([x, y])

    def set_goal(self, x, y):
        """
        Sets the goal position.
        """
        self.goal = np.array([x, y])

    def set_limits(self, x_min, x_max, y_min, y_max):
        """
        Sets the limits for the x and y coordinates. These values are used by
        the PSO as lower and upper boundaries of the search space.
        """
        self.limits = np.array([x_min, x_max, y_min, y_max])

    def add_circle(self, circle_idx, x=0.0, y=0.0, r=1.0, Kv=100.0):
        """
        Adds a circular obstacle.

        x, y        centroid (center)
        r           radius
        Kv          scaling factor
        """
        self.obs.setdefault('Circle',[])
        data = (x, y, r, Kv)
        if circle_idx < len(self.obs['Circle']):
            self.obs['Circle'][circle_idx] = data
        else:
            self.obs['Circle'].extend([None]*(circle_idx-len(self.obs['Circle'])+1))
            self.obs['Circle'][circle_idx] = data

    def add_ellipse(self, x=0.0, y=0.0, theta=0.0, a=0.0, b=0.0, Kv=100.0):
        """
        Adds an elliptical obstacle.

        x, y        centroid (center)
        theta       rotation (angle between semi-major axis and x-axis)
        a           semi-major axis
        b           semi-minor axis
        Kv          scaling factor
        """
        e = np.sqrt(1.0 - b ** 2 / a ** 2)          # Eccentricity
        data = ("Ellipse", x, y, theta, b, e, Kv)
        self.obs.append(data)

    def add_convex(self, V, Kv=100.0):
        """
        Adds a convex polygonal obstacle.

        x, y        centroid
        V           vertices (each row is an x-y pair)
        Kv          scaling factor
        """
        V = np.asarray(V)
        x, y = centroid(V)
        data = ("Convex", x, y, V, Kv)
        self.obs.append(data)

    def add_polygon(self, V, center=None, Kv=100.0):
        """
        Adds a polygonal obstacle.

        x, y        centroid
        V           vertices (each row is an x-y pair)
        Kv          scaling factor
        """
        V = np.asarray(V)
        x, y = centroid(V)
        data = ("Polygon", x, y, V, Kv)
        self.obs.append(data)

    def remove_obs(self, name, idx):
        """
        Removes an obstacle from the list.
        """
        self.obs.get(name)[idx] = None


    def optimize(self, nPts=3, ns=100, nPop=40, epochs=500, K=0, phi=2.05,
                 vel_fact=0.5, conf_type='MX', IntVar=None, normalize=False,
                 rad=0.1, f_interp='cubic', Xinit=None):
        """
        Optimizes the path.
        """
        # Arguments passed to the function to minimize (<args> has five items)
        Xs = np.ones((nPop, 1)) * self.start[0]   # Start x-position (as array)
        Ys = np.ones((nPop, 1)) * self.start[1]   # Start y-position (as array)
        Xg = np.ones((nPop, 1)) * self.goal[0]    # Goal x-position (as array)
        Yg = np.ones((nPop, 1)) * self.goal[1]    # Goal y-position (as array)
        args = [(Xs, Ys), (Xg, Yg),  self.obs, ns, f_interp]

        # Boundaries of the search space
        nVar = 2 * nPts
        # for each sampled spline searching position x y limits
        LB = np.zeros(nVar)
        UB = np.zeros(nVar)
        LB[:nPts] = self.limits[0]
        UB[:nPts] = self.limits[1]
        LB[nPts:] = self.limits[2]
        UB[nPts:] = self.limits[3]

        # Optimize
        X, info = PSO(fitness, LB, UB, nPop, epochs, K, phi, vel_fact,
                      conf_type, IntVar, normalize, rad, args, Xinit)

        # Get the results for the best path (<args> has six items)
        args = [self.start, self.goal,  self.obs, ns, f_interp, []]
        F = fitness(X.reshape(1, nVar), args)
        L, count, Px, Py = args[5]
        self.sol = (X, L[0], count, Px, Py)

    def plot_obs(self, ax):
        """
        Plots the obstacles.

        Legend:
        obstacle centroids   -->   orange x markers
        obstacles            -->   wheat colored objects
        """

        element = None
        self.obs.setdefault('Circle', [])
        self.obs.setdefault('Ellipse', [])
        self.obs.setdefault('Polygon', [])
        self.obs.setdefault('Convex', [])

        for name, obs_set in self.obs.items():
            # Obstacle data
            for data in obs_set:
            # Obstacle type and its centroid
                if data == None:
                    break

                xc, yc = data[:2]

                # Obstacle is a circle (r=radius)
                if (name == 'Circle'):
                    r = data[2]
                    element = Circle((xc, yc), r, fc='wheat', ec=None)

                # Obstacle is an ellipse (theta=rotation from x-axis, b=semi-minor
                # axis, e=eccentricity)
                elif (name == 'Ellipse'):
                    theta, b, e = data[2:5]
                    theta = theta * 180.0 / np.pi
                    b = 2 * b                           # Minor axis
                    a = b / np.sqrt(1.0 - e ** 2)       # Major axis
                    element = Ellipse((xc, yc), a, b, theta, fc='wheat', ec=None)

                # Obstacle is a convex polygon (V=vertices)
                elif (name == 'Convex'):
                    V = data[2]
                    element = Polygon(V, closed=True, fc='wheat', ec=None)

                # Obstacle is a convex polygon (V=vertices)
                elif (name == 'Polygon'):
                    V = data[2]
                    element = Polygon(V, closed=True, fc='wheat', ec=None)

                ax.add_patch(element)                   # Add element to the plot
                ax.plot(xc, yc, 'x', ms=4, c='orange')  # Add centroid position

        # Plot only what is inside the limits
        ax.set_xlim(self.limits[0], self.limits[1])
        ax.set_ylim(self.limits[2], self.limits[3])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_path(self, ax):
        """
        Plots the obstacles.

        Legend:
        start position         -->   black circle marker
        goal position          -->   black star marker
        path                   -->   red line
        internal breakpoints   -->   blue point markers
        """
        # Coordinates of the discretized path
        Px = self.sol[3]
        Py = self.sol[4]

        # Plot the spline
        ax.plot(Px[0, :], Py[0, :], lw=0.50, c='r')

        # Plot the internal breakpoints
        X = self.sol[0]
        nPts = len(X) // 2
        ax.plot(X[:nPts], X[nPts:], '.', ms=4, c='b')

        # Plot start position
        ax.plot(self.start[0], self.start[1], 'o', ms=6, c='k')

        # Plot goal position
        ax.plot(self.goal[0], self.goal[1], '*', ms=8, c='k')
