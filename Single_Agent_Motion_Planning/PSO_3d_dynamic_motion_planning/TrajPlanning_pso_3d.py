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

# initialize the trajectory control variables
def build_Ctrl_init(n_steps, numControls):
    max_uxy = 8.0
    max_uz = 4.0

    control = np.zeros((n_steps, numControls))
    control[:, 0] = np.random.uniform(0, max_uxy, n_steps)  # uxy
    control[:, 1] = np.random.uniform(0, 2 * np.pi, n_steps)  # theta
    control[:, 2] = np.random.uniform(-max_uz, max_uz, n_steps)  # uz

    Xinit = np.concatenate((control[:,0], control[:,1], control[:,2]))

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


def cal_collision_penalty(obs, Px, Py, Pz, numFrames):
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
        for frame, obs_set in obs_set.items():
            if frame==0:
                continue
            # skip the start frame
            for data in obs_set:
                # Obstacle type is ball and its centroid
                # Obstacle is a 3d ball (r = radius, Kv = scaling factor)
                if (name == 'Ball'):
                    xc, yc, zc = data[:3]
                    # Distances from the obstacle centroid
                    # d = np.sqrt((Px - xc) ** 2 + (Py - yc) ** 2 + (Pz - zc) ** 2)
                    d = np.sqrt((Px[:, frame] - xc) ** 2 + (Py[:, frame] - yc) ** 2 + (Pz[:, frame] - zc) ** 2)
                    r, Kv = data[3:]
                    inside = r * 2 > d
                    # inside = 1.45 > d

                # Obstacle is a circle (r = radius, Kv = scaling factor)
                elif (name == 'Circle'):
                    # Obstacle type and its centroid
                    xc, yc = data[:2]
                    # Distances from the obstacle centroid
                    d = np.sqrt((Px - xc) ** 2 + (Py - yc) ** 2)
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
                # err += np.nanmean(penalty, axis=1)
                err +=penalty

    return err, count


# check trajectories feasibility
def feasibility_check(Uxy, Uz, Uxyz, Vxy, Vz, Vxyz):
    feasible = True
    infeasibility_count = 0
    max_vxyz = 8.0  # [m/s]
    max_vxy = 6.0
    max_vz = 3.0
    max_uxyz = 8.0  # [m/s^2]
    max_uxy = 8.0
    max_uz = 4.0

    uxy_infeasible = np.abs(Uxy) - max_uxy > 0
    uz_infeasible = np.abs(Uz) - max_uz > 0
    uxyz_infeasible = np.abs(Uxyz) - max_uxyz > 0
    vxy_infeasible = np.abs(Vxy) - max_vxy > 0
    vz_infeasible = np.abs(Vz) - max_vz > 0
    vxyz_infeasible = np.abs(Vxyz) - max_vxyz > 0

    if (uxy_infeasible.any() or uz_infeasible.any() or uxyz_infeasible.any() or vxy_infeasible.any() or vz_infeasible.any() or vxyz_infeasible.any()):
        feasible = False
        infeasibility_count += 1

    return feasible, infeasibility_count


def fitness(X, args):
    """
    Returns the function to minimize, i.e. the path length when there is
    not any obstacle violation.

    The interpolation method can be "slinear", "quadratic", or "cubic" (spline
    of order 1, 2, and 3, respectively). The curvilinear coordinate along the
    path is taken in the interval from 0 (start) to 1 (goal).
    """
    # Arguments passed
    Xs, Ys, Zs = args[0]            # Start position (as array)
    Xg, Yg, Zg = args[1]            # Goal position (as array)
    obs = args[2]                   # List of obstacles
    ns, nCtrls = args[3]                    # Number of points along the path
    f_interp = args[4]              # Interpolation method: 'slinear' is preferred for its convergence and efficiency
    time_step = 0.1
    max_vxyz = 8.0  # [m/s]
    max_vxy = 6.0
    max_vz = 3.0
    min_vz = -3.0
    max_uxyz = 8.0  # [m/s^2]
    max_uxy = 8.0
    max_uz = 4.0

    # smooth_penalty = np.zeros(args[0][0].shape[0])
    # jerk_penalty = np.zeros(args[0][0].shape[0])

    w_l = 5   # path length penalty weight
    w_g = 20.0   # path goal penalty weight
    w_c = 10.0
    w_s = 5
    w_f = 100

    nPop, nVar = X.shape
    nframes = nVar // nCtrls            # Number of (internal) breakpoints

    # Coordinates of the breakpoints (start + internal + goal)
    ctrl_uxy = X[:,:nframes]
    ctrl_thetaxy = X[:,nframes:2*nframes]
    ctrl_uz = X[:,2*nframes:]

    # Classes defining the spline
    t = np.linspace(0, 1, nframes)
    CS_uxy = interp1d(t, ctrl_uxy, axis=1, kind=f_interp, assume_sorted=True)
    CS_thetaxy = interp1d(t, ctrl_thetaxy, axis=1, kind=f_interp, assume_sorted=True)
    CS_uz = interp1d(t, ctrl_uz, axis=1, kind=f_interp, assume_sorted=True)

    # Coordinates of the discretized path
    s = np.linspace(0, 1, ns)

    Uxy = CS_uxy(s)
    Theta_xy = CS_thetaxy(s)
    Ux = Uxy * np.cos(Theta_xy)
    Uy = Uxy * np.sin(Theta_xy)
    Uz = CS_uz(s)
    Uxyz = np.linalg.norm(np.stack((Ux, Uy, Uz), axis=-1), axis=-1)

    Vx = np.zeros((nPop, nframes+1))
    Vy = np.zeros((nPop, nframes+1))
    Vz = np.zeros((nPop, nframes+1))
    for i in range(1,nframes+1):
        Vx[:, i] = Vx[:, i - 1] + Ux[:, i - 1] * time_step
        Vy[:, i] = Vy[:, i - 1] + Uy[:, i - 1] * time_step
        Vz[:, i] = Vz[:, i - 1] + Uz[:, i - 1] * time_step
    Vxy = np.linalg.norm(np.stack((Vx, Vy), axis=-1), axis=-1)
    Vxyz = np.linalg.norm(np.stack((Vx, Vy, Vz), axis=-1), axis=-1)

    _, infeasibility_count = feasibility_check(Uxy, Uz, Uxyz, Vxy, Vz, Vxyz)

    Px = np.block([Xs, np.zeros((nPop, nframes))])
    Py = np.block([Ys, np.zeros((nPop, nframes))])
    Pz = np.block([Zs, np.zeros((nPop, nframes))])
    # Px[:,1:] = Px[:,:-1] + Vx[:,:-1] * time_step
    # Py[:, 1:] = Py[:, :-1] + Vy[:, :-1] * time_step
    # Pz[:, 1:] = Pz[:, :-1] + Vz[:, :-1] * time_step
    for i in range(1,nframes+1):
        Px[:, i] = Px[:, i-1] + Vx[:, i-1] * time_step + 0.5 * Ux[:, i-1] * time_step ** 2
        Py[:, i] = Py[:, i-1] + Vy[:, i-1] * time_step + 0.5 * Uy[:, i-1] * time_step ** 2
        Pz[:, i] = Pz[:, i-1] + Vz[:, i-1] * time_step + 0.5 * Uz[:, i-1] * time_step ** 2

    # get states and control trajectories
    dX = np.diff(Px[:,:-1], axis=1)
    dY = np.diff(Py[:,:-1], axis=1)
    dZ = np.diff(Pz[:,:-1], axis=1)

    # calculate Path length
    L = np.sqrt(dX ** 2 + dY ** 2 + dZ ** 2).sum(axis=1)

    # calculate Goal penalty value
    current_position = np.transpose(np.array([Px[:,-1], Py[:,-1], Pz[:,-1]]))
    goal = np.hstack((Xg, Yg, Zg))
    goal_penalty = np.linalg.norm(current_position - goal, axis=1)

    # calculate Collision penalty value
    collision_penalty, collision_count = cal_collision_penalty(obs, Px, Py, Pz, nframes)

    # Calculate Smooth penalty value
    Ux_jerk = np.diff(Ux, axis=1)
    Uy_jerk = np.diff(Uy, axis=1)
    Uz_jerk = np.diff(Uz, axis=1)
    smooth_penalty = np.sum(Ux_jerk ** 2 + Uy_jerk ** 2 + Uz_jerk ** 2, axis=1)

    # Calculate Jerk penalty value
    jerk_penalty = np.sum(np.diff(Ux_jerk) ** 2 + np.diff(Uy_jerk) ** 2 + np.diff(Uz_jerk), axis=1)

    # calculate Feasiblity penalty value
    acceleration_xy_penalty = np.sum(np.maximum(0, np.abs(Uxy) - max_uxy) ** 2, axis=1)
    velocity_xy_penalty = np.sum(np.maximum(0, np.abs(Vxy) - max_vxy) ** 2, axis=1)
    acceleration_xyz_penalty = np.sum(np.maximum(0, np.abs(Uxyz) - max_uxyz) ** 2, axis=1)
    velocity_xyz_penalty = np.sum(np.maximum(0, np.abs(Vxyz) - max_vxyz) ** 2, axis=1)
    acceleration_z_penalty = np.sum(np.maximum(0, np.abs(Uz)-max_uz)**2, axis=1)
    velocity_z_penalty = np.sum(np.maximum(0, np.abs(Vz)-max_vz) ** 2, axis=1)

    # Function to minimize
    fitness = L * (w_l + collision_penalty) + w_g * goal_penalty + w_s * (smooth_penalty + jerk_penalty) + w_f * (acceleration_xy_penalty + velocity_xy_penalty + acceleration_xyz_penalty + velocity_xyz_penalty+acceleration_z_penalty+velocity_z_penalty)
    # fitness = w_l*L + w_g * goal_penalty + w_c * collision_penalty + w_s * (smooth_penalty) + w_f * (acceleration_penalty + velocity_penalty)

    # Return the results for the best path if it is the last call
    if (len(args) == 6):
        args[5] = [L, collision_count, infeasibility_count, Px, Py, Pz]

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


    def set_start(self, x, y, z):
        """
        Sets the start position.
        """
        self.start = np.array([x, y, z])


    def set_goal(self, x, y, z):
        """
        Sets the goal position.
        """
        self.goal = np.array([x, y, z])


    def set_limits(self, x_min, x_max, y_min, y_max, z_min, z_max):
        """
        Sets the limits for the x and y coordinates. These values are used by
        the PSO as lower and upper boundaries of the search space.
        """
        self.limits = np.array([x_min, x_max, y_min, y_max, z_min, z_max])


    def add_ball(self, frame_idx, ball_idx, x=0.0, y=0.0, z=0.0, r=1.0, Kv=100.0):
        """
        Adds a ball obstacle.

        x, y, z        centroid (center)
        r           radius
        Kv          scaling factor
        """
        self.obs.setdefault('Ball',{})
        data = (x, y, z, r, Kv)

        if frame_idx not in self.obs['Ball']:
            self.obs['Ball'][frame_idx] = []

        if ball_idx < len(self.obs['Ball'][frame_idx]):
            self.obs['Ball'][frame_idx][ball_idx] = data
        else:
            self.obs['Ball'][frame_idx].extend([None]*(ball_idx-len(self.obs['Ball'][frame_idx])+1))
            self.obs['Ball'][frame_idx][ball_idx] = data


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


    def optimize(self, nCtrls=3, ns=100, nPop=40, epochs=500, K=0, phi=2.05,
                 vel_fact=0.5, conf_type='RB', IntVar=None, normalize=False,
                 rad=0.1, f_interp='cubic', Xinit=None):
        """
        Optimizes the path.
        """
        # Arguments passed to the function to minimize (<args> has five items)
        Xs = np.ones((nPop, 1)) * self.start[0]   # Start x-position (as array)
        Ys = np.ones((nPop, 1)) * self.start[1]   # Start y-position (as array)
        Zs = np.ones((nPop, 1)) * self.start[2]  # Start z-position (as array)
        Xg = np.ones((nPop, 1)) * self.goal[0]    # Goal x-position (as array)
        Yg = np.ones((nPop, 1)) * self.goal[1]    # Goal y-position (as array)
        Zg = np.ones((nPop, 1)) * self.goal[2]    # Goal z-position (as array)
        args = [(Xs, Ys, Zs), (Xg, Yg, Zg), self.obs, (ns,nCtrls), f_interp]

        # Boundaries of the search space
        nVar = len(Xinit)
        nFrames = nVar // nCtrls
        max_uxy = 8.0
        max_uz = 4.0

        # for each optimized control u_xy, theta_xy, uz limits
        LB = np.hstack(( 0*np.ones((1, nFrames)), 0*np.ones((1, nFrames)), -max_uz*np.ones((1, nFrames)) ))
        UB = np.hstack(( max_uxy * np.ones((1, nFrames)), 2*np.pi * np.ones((1, nFrames)), max_uz * np.ones((1, nFrames)) ))

        # Optimize
        X, info = PSO(fitness, LB, UB, nPop, epochs, K, phi, vel_fact,
                      conf_type, IntVar, normalize, rad, args, Xinit)

        # Get the results for the best path (<args> has six items)
        args = [self.start, self.goal, self.obs, (ns, nCtrls), f_interp, []]
        F = fitness(X.reshape(1, nVar), args)       # this final interpolation is very important to generate solution
        L, obs_violation, feasibility_violation, Px, Py, Pz = args[5]
        self.sol = (X, L[0], obs_violation, feasibility_violation, Px, Py, Pz)


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
        self.obs.setdefault('Ball',[])

        for name, obs_set in self.obs.items():
            # Obstacle data
            for data in obs_set:
            # Obstacle type and its centroid
                if data == None:
                    break

                xc, yc, zc = data[:3]

                # Obstacle is a ball (r=radius)
                if (name == 'Ball'):
                    r = data[3]
                    element = Ball((xc, yc, zc), r, ball_color='wheat')

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
                ax.plot(xc, yc, zc, 'x', ms=4, c='orange')  # Add centroid position

        # Plot only what is inside the limits
        ax.set_xlim(self.limits[0], self.limits[1])
        ax.set_ylim(self.limits[2], self.limits[3])
        ax.set_zlim(self.limits[4], self.limits[5])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


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
        Pz = self.sol[5]

        # Plot the spline
        ax.plot(Px[0, :], Py[0, :], Pz[0, :], lw=0.50, c='r')

        # Plot the internal breakpoints
        X = self.sol[0]
        nPts = len(X) // 2
        ax.plot(X[:nPts], X[nPts:2*nPts], X[2*nPts:], '.', ms=4, c='b')

        # Plot start position
        ax.plot(self.start[0], self.start[1], self.start[2],'o', ms=6, c='k')

        # Plot goal position
        ax.plot(self.goal[0], self.goal[1], self.goal[2], '*', ms=8, c='k')

class Ball:

    def __init__(self, ball_center, radius, ball_color='r', **kwargs):
        self.center = np.array(ball_center)
        self.radius = radius
        self.ball_color = ball_color
        self.kwargs = kwargs

    def plot(self, ax):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        u, v = np.meshgrid(u, v)
        x = self.center[0] + self.radius * np.outer(np.cos(u), np.sin(v))
        y = self.center[1] + self.radius * np.outer(np.sin(u), np.sin(v))
        z = self.center[2] + self.radius * np.outer(np.ones(np.size(u)), np.cos(v))
        self.surface = ax.plot_surface(x, y, z, **self.kwargs, color=self.ball_color, alpha = 0.1)
        self.obs_center = ax.plot(self.center[0], self.center[1], self.center[2], 'x', ms=4, c='orange')[0]

    def update(self, ax, new_center):
        self.center = new_center
        self.surface.remove()  # Remove the old surface
        self.obs_center.remove()
        self.plot(ax)  # Plot the new surface
