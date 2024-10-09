"""
Path Planning Using Particle Swarm Optimization


Implementation of particle swarm optimization (PSO) for path planning when the
environment is known.


Copyright (c) 2021 Gabriele Gilardi


Features
--------
- The code has been written and tested in Python 3.8.5.
- Four types of obstacles: circle, ellipse, convex polygon, generic polygon.
- Start position, goal position, and obstacles can be dynamically changed to
  simulate motion.
- Penalty function of type 1/d with the center in the obstacle centroid.
- To improve the execution speed, the algorithms to determine if a point is
  inside an obstacle have been designed to carry out the determination on all
  points simultaneously.
- Points on the obstacle borders/edges are not considered inside the obstacle.
- Option to run sequential tests with different initial conditions to increase
  the chances to find a global minimum.
- Usage: python test.py <example>.

Main Parameters
---------------
example
    Number of the example to run (1, 2, or 3.)
start
    Start coordinates.
goal
    Goal coordinates.
limits
    Lower and upper boundaries of the map and search space in the PSO.
nRun
    Number of runs.
nPts >= 2
    Number of internal points defining the spline. The number of variables is
    twice this number.
d >= 2
    Number of segments between the spline breakpoints.
nPop >=1, epochs >= 1
    Number of agents (population) and number of iterations.
f_interp = slinear, quadratic, cubic
    Order of the spline (1st, 2nd and 3rd order, respectively.)
Xinit
    Initial value of the variables. Set Xinit=None to pick them randomly. This
    array is organized with first the x-coordinates of all internal points and
    then the y-coordinates of all internal points.
K >= 0
    Average size of each agent's group of informants. If K=0 the entire swarm
    is used as agent's group of informants.
phi >= 2
    Coefficient to calculate the self-confidence and the confidence-in-others
    coefficients.
vel_fact > 0
    Velocity factor to calculate the max. and min. allowed velocities.
conf_type = HY, RB, MX
    Confinement on the velocities: hyperbolic, random-back, mixed.
IntVar
    List of indexes specifying which variable should be treated as integer.
    If all variables are real set IntVar=None, if all variables are integer
    set IntVar='all'.
normalize = True, False
    Specifies if the search space should be normalized.
0 < rad < 1
    Normalized radius of the hypersphere centered on the best particle. The
    higher the number of other particles inside and the better is the solution.

Examples
--------
There are three examples, all of them using the same obstacles:

Example 1
- Multiple runs, cubic spline, optimizer initialized randomly.
- See <Results_Figure_1.png> for the full results.

Example 2
- Multiple runs, quadratic spline, optimizer initialized with the straight line
  between start and goal position.
- See <Results_Example_2.png> for the full results.

Example 3
- Single run, linear spline, optimizer initialized with the previous solution,
  start point chasing a moving goal with one obstacle (the circle) also moving.
- See <Results_Example_3.gif> for the full animation.

References
----------
- PSO: https://github.com/gabrielegilardi/PSO.git.
- Centroid calculation: http://en.wikipedia.org/wiki/Centroid.
- Points inside polygons: http://paulbourke.net/geometry/polygonmesh/.
"""

import sys
from copy import deepcopy
import numpy as np
from matplotlib.animation import FuncAnimation
import time

from PathPlanning_pso_3d import *
# from TrajPlanning_pso_3d import *

def main():
    example = 'dynamic_ex'
    print('PSO planning Start!')
    # Define start, goal, and limits
    start = (0, 5, 6)
    goal = (2, -5, -6)
    limits = [-2, 10, -6, 6, -15, 15]
    layout = PathPlanning(start, goal, limits)

    # Add obstacles (polygon verteces must be given in counter-clockwise order)
    # layout.add_ellipse(x=5, y=-1, theta=-np.pi/6, a=1.0, b=0.5, Kv=100)
    # V = [(2, 1), (5, 1), (5, 5), (4, 5), (2, 4)]
    # layout.add_convex(V, Kv=100)
    # V = [(6.5, -1), (9.5, -1), (9.5, 4), (8.5, 3), (8.5, 0), (7.5, 0),
    #      (7.5, 3), (6.5, 3)]
    # layout.add_polygon(V, Kv=100)
    layout.add_ball(0, x=2, y=-2, z=2, r=0.75, Kv=100)
    layout.add_ball(1, x=4, y=-4, z=-2, r=0.75, Kv=100)

    # Example moving obstacles: single run, linear spline, optimizer initialized with the
    # previous solution, start point chasing a fixed goal with two obstacle
    # (the ball) also moving.
    # No obstacle violations.
    # See <Results_Example_3.gif> for the full animation.
    if (example == 'dynamic_ex'):

        nRun = 65
        nPts = 5
        # d = nRun                  # ns = 301 (points along the spline)
        nPop = 100
        epochs = 200
        f_interp = 'slinear'
        Xinit = None            # Re-defined after each run
        # Xinit = build_Xinit(layout.start, layout.goal, nPts)  # init as the extracted controls in replan parameters
        t0 = time.time()

    else:
        print("Example not found")
        sys.exit(1)

    # Init other parameters
    np.random.seed(1294404794)
    np.seterr(all='ignore')
    # ns = 1 + (nPts + 1) * d         # Number of points along the spline
    best_L = np.inf                 # Best length (minimum)
    best_run = 0                    # Run corresponding to the best length
    best_count = 0                  # count corresponding to the best length
    paths = [None] * nRun           # List with the results from all runs
    d = nRun                        # ns = 301 (points along the spline)
    d1 = d

    # Run cases

    for run in range(nRun):
        ns = 1 + (nPts + 1) * d  # Number of points along the spline
        # print("\nns = ", ns)
        # Optimize (the other PSO parameters have always their default values)
        layout.optimize(nPts=nPts, ns=ns, nPop=nPop, epochs=epochs,
                        f_interp=f_interp, Xinit=Xinit)

        # Save run
        paths[run] = deepcopy(layout)

        # Print results
        L = layout.sol[1]               # Length
        count = layout.sol[2]           # Number of violated obstacles
        print("\nrun={0:d}, L={1:.2f}, count={2:d}"
              .format(run+1, L, count), end='', flush=True)

        # Save if best result (regardless the violations)
        if (L < best_L):
            best_L = L
            best_run = run
            best_count = count

        # Only for example 3 (move start, goal, and circular obstacle)
        if (example == 'dynamic_ex'):

            # Print the current start, goal, and circle centroid
            # print(", s=[{0:.2f},{1:.2f},{2:.2f}], g=[{3:.2f},{4:.2f},{5:.2f}], c1=[{6:.2f},{7:.2f},{8:.2f}], c2=[{9:.2f},{10:.2f},{11:.2f}]"
            #       .format(layout.start[0], layout.start[1], layout.start[2], layout.goal[0],
            #       layout.goal[1], layout.goal[2], layout.obs.get('Ball')[0][0], layout.obs.get('Ball')[0][1], layout.obs.get('Ball')[0][2], layout.obs.get('Ball')[1][0], layout.obs.get('Ball')[1][1],  layout.obs.get('Ball')[1][2] ), end='',
            #       flush=True)   # , layout.obs[1][1], layout.obs[1][2] ,c2=[{6:.2f},{7:.2f}]

            # Path coordinates
            Px = layout.sol[3].flatten()
            Py = layout.sol[4].flatten()
            Pz = layout.sol[5].flatten()

            # Move the start position along the tangential direction of the
            # current path with a"speed" of 0.1
            # vel_s = 0.1
            # theta_s = np.arctan2(Py[1]-Py[0], Px[1]-Px[0])
            # x_s = layout.start[0] + vel_s * np.cos(theta_s)
            # y_s = layout.start[1] + vel_s * np.sin(theta_s)

            if run + 1 < nRun:
                x_s = Px[3 + nPts]
                y_s = Py[3 + nPts]
                z_s = Pz[3 + nPts]
            else:
                x_s = (Px[-1] + goal[0])/2
                y_s = (Py[-1] + goal[1])/2
                z_s = (Pz[-1] + goal[2])/2

            # x_s = Px[1 + nPts]
            # y_s = Py[1 + nPts]
            # z_s = Pz[1 + nPts]
            layout.set_start(x_s, y_s, z_s)

            # Move the goal position along a straight line with a "speed" of 0.15
            vel_g = 0.0
            theta_g = -165.0 * np.pi / 180.0
            x_g = layout.goal[0] + vel_g * np.cos(theta_g)
            y_g = layout.goal[1] + vel_g * np.sin(theta_g)
            z_g = layout.goal[2]
            layout.set_goal(x_g, y_g, z_g)

            # Move the circular obstacle along a straight line with a "speed" of 0.1
            vel_o1 = 0.2
            vel_o2 = 0.3

            if (run > 25):
                vel_o1 = -vel_o1          # Invert direction after 25 runs
            if run > 45:
                vel_o2 = -vel_o2
            alpha_o = -60 * np.pi / 180.0
            theta_o = +150.0 * np.pi / 180.0
            x_o1 = layout.obs.get('Ball')[0][0] + vel_o1 * np.cos(alpha_o) * np.cos(theta_o)
            y_o1 = layout.obs.get('Ball')[0][1] + vel_o1 * np.cos(alpha_o) * np.sin(theta_o)
            z_o1 = layout.obs.get('Ball')[0][2] + vel_o1 * np.sin(alpha_o)
            layout.remove_obs('Ball', 0)
            layout.add_ball(0, x_o1, y_o1, z_o1, r=0.75, Kv=100)

            x_o2 = layout.obs.get('Ball')[1][0] + vel_o2 * np.cos(alpha_o) * np.cos(theta_o)
            y_o2 = layout.obs.get('Ball')[1][1] + vel_o2 * np.cos(alpha_o) * np.sin(theta_o)
            z_o2 = layout.obs.get('Ball')[1][2] + vel_o2 * np.sin(alpha_o)
            layout.remove_obs('Ball', 1)
            layout.add_ball(1, x_o2, y_o2, z_o2, r=0.75, Kv=100)

            # Initialize the solution with the last optimal path
            Xinit = layout.sol[0]
            d -= 1  # ns also updated


    # Animation for example
    if (example == 'dynamic_ex'):
        tf = time.time()
        print("\nPSO solving time: ", tf - t0)

        def animate(row):

            # Path x and y coordinates
            path.set_xdata(Px_set[row])
            path.set_ydata(Py_set[row])
            # path.set_zdata(Pz_set[row])
            path.set_3d_properties(Pz_set[row])

            # Start position x and y coordinates
            Ps.set_xdata(start[row, 0])
            Ps.set_ydata(start[row, 1])
            # Ps.set_zdata(start[row, 2])
            Ps.set_3d_properties(start[row, 2])

            # Goal position x and y coordinates
            Pg.set_xdata(goal[row, 0])
            Pg.set_ydata(goal[row, 1])
            # Pg.set_zdata(goal[row, 2])
            Pg.set_3d_properties(goal[row, 2])

            # Text with the path length
            L_text.set_text(lengths[row])
            # L_text.set_text(f'Path length: {lengths[row]:.2f}')

            # Obstacle and its centroid
            obs1.update(ax, new_center=(XYZc1[row, 0], XYZc1[row, 1], XYZc1[row, 2]))
            # XYZobs1.set_xdata(XYZc1[row, 0])
            # XYZobs1.set_ydata(XYZc1[row, 1])
            # # XYZobs1.set_zdata(XYZc1[row, 2])
            # XYZobs1.set_3d_properties(XYZc1[row, 2])


            obs2.update(ax, new_center=(XYZc2[row, 0], XYZc2[row, 1], XYZc2[row, 2]))
            # XYZobs2.set_xdata(XYZc2[row, 0])
            # XYZobs2.set_ydata(XYZc2[row, 1])
            # # XYZobs2.set_zdata(XYZc2[row, 2])
            # XYZobs2.set_3d_properties(XYZc1[row, 2])

            # return path, Ps, Pg, L_text, XYZobs1, obs1.surface

        start = np.zeros((nRun, 3))  # Start position
        goal = np.zeros((nRun, 3))  # Goal position
        lengths = [None] * nRun  # Length (text)
        XYZc1 = np.zeros((nRun, 3))  # Obstacle1 (ball) centroid
        XYZc2 = np.zeros((nRun, 3))  # Obstacle2 (ball) centroid
        Px_set = []
        Py_set = []
        Pz_set = []

        # Build the arrays needed for the animation
        for run in range(nRun):
            # Elements to be animated
            ns = 1 + (nPts + 1) * d1
            Px = np.zeros((1, ns))  # Path x coordinates
            Py = np.zeros((1, ns))  # Path y coordinates
            Pz = np.zeros((1, ns))  # Path z coordinates

            layout = paths[run]
            Px_set.append(layout.sol[3].flatten())
            Py_set.append(layout.sol[4].flatten())
            Pz_set.append(layout.sol[5].flatten())
            start[run, :] = layout.start
            goal[run, :] = layout.goal
            lengths[run] = "run=" + str(run+1) \
                           + ", L=" + str("{:.2f}".format(layout.sol[1]))
            XYZc1[run, :] = layout.obs.get('Ball')[0][0:3]
            XYZc2[run, :] = layout.obs.get('Ball')[1][0:3]

            d1 -= 1

        # Plot obstacles (except circle)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, aspect='equal', projection="3d")    # Create a 3D plot
        paths[0].remove_obs('Ball', 0)
        paths[0].remove_obs('Ball', 1)

        # Plot only what is inside the limits
        ax.set_xlim(paths[0].limits[0], paths[0].limits[1])
        ax.set_ylim(paths[0].limits[2], paths[0].limits[3])
        ax.set_zlim(paths[0].limits[4], paths[0].limits[5])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        # Path
        path = ax.plot(Px_set[0], Py_set[0], Pz_set[0], lw=0.5, c='b')[0]

        # Start and goal
        Ps = ax.plot(start[0, 0], start[0, 1], start[0, 2], 'o', ms=6, c='k')[0]
        Pg = ax.plot(goal[0, 0], goal[0, 1], goal[0, 2], '*', ms=8, c='k')[0]

        # Length (text)
        xt = paths[0].limits[0] + 0.05 * (paths[0].limits[1] - paths[0].limits[0])
        yt = paths[0].limits[2] + 0.05 * (paths[0].limits[3] - paths[0].limits[2])
        zt = paths[0].limits[4] + 0.05 * (paths[0].limits[5] - paths[0].limits[4])
        L_text = ax.text(xt, yt, zt, lengths[0], fontsize=10)

        # Obstacle (ball)
        r = 0.75
        obs1 = Ball(ball_center=(XYZc1[0, 0], XYZc1[0, 1], XYZc1[0, 2]), radius=r, ball_color='wheat')
        obs1.plot(ax)
        # XYZobs1 = ax.plot(XYZc1[0, 0], XYZc1[0, 1], XYZc1[0, 2], 'x', ms=4, c='orange')[0]

        obs2 = Ball(ball_center=(XYZc2[0, 0], XYZc2[0, 1], XYZc2[0, 2]), radius=r, ball_color='wheat')
        obs2.plot(ax)
        # XYZobs2 = ax.plot(XYZc2[0, 0], XYZc2[0, 1], XYZc2[0, 2], 'x', ms=4, c='orange')[0]

        # Animate and save a copy
        anim = FuncAnimation(fig, animate, interval=200, frames=nRun-1, blit=False)
        print("\n")
        # anim.save("Results_Example.gif")
        plt.show()


if __name__ == '__main__':
    main()