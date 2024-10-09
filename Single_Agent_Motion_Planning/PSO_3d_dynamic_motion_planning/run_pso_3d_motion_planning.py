"""
Path Planning Using Particle Swarm Optimization


Implementation of particle swarm optimization (PSO) for path planning


Features
--------
- The code has been written and tested in Python 3.11.
- Four types of obstacles: circle, ellipse, convex polygon, generic polygon.
- Start position, goal position, and obstacles can be dynamically changed to
  simulate motion.
- Penalty function of type 1/d with the center in the obstacle centroid.
- To improve the execution speed, the algorithms to determine if a point is
  inside an obstacle have been designed to carry out the determination on all
  points simultaneously.


References
----------
- PSO: https://github.com/gabrielegilardi/PSO.git.

"""

import sys
from copy import deepcopy
import numpy as np
from matplotlib.animation import FuncAnimation
import time
from TrajPlanning_pso_3d import *

def main():
    example = 'dynamic_ex'
    print('PSO planning Start!')
    # Define start, goal, and limits
    t0 = time.time()
    start = (0, 5, 6)
    goal = (2, -5, -6)
    limits = [-2, 10, -6, 6, -15, 15]       # environment limits
    layout = PathPlanning(start, goal, limits)

    # Add obstacles (polygon verteces must be given in counter-clockwise order)
    # layout.add_ellipse(x=5, y=-1, theta=-np.pi/6, a=1.0, b=0.5, Kv=100)
    # V = [(2, 1), (5, 1), (5, 5), (4, 5), (2, 4)]
    # layout.add_convex(V, Kv=100)
    # V = [(6.5, -1), (9.5, -1), (9.5, 4), (8.5, 3), (8.5, 0), (7.5, 0),
    #      (7.5, 3), (6.5, 3)]
    # layout.add_polygon(V, Kv=100)
    layout.add_ball(0, 0, x=2, y=-2, z=2, r=0.75, Kv=100)
    layout.add_ball(0, 1, x=4, y=-4, z=-2, r=0.75, Kv=100)

    # Example moving obstacles: single run, linear spline, optimizer initialized with the
    # previous solution, start point chasing a fixed goal with two obstacle
    # (the ball) also moving.
    # No obstacle violations.
    # See <Results_Example.gif> for the full animation.
    if (example == 'dynamic_ex'):

        nRun = 2
        nFrames = 65
        nCtrls = 3                  # control variables: uxy, xy_angle, uz
        nPop = 100
        ns = 1 + nCtrls * nFrames*2  # Number of points along the path
        epochs = 200
        f_interp = 'slinear'
        Xinit = build_Ctrl_init(nFrames, nCtrls)  # init as the extracted controls in replan parameters

    else:
        print("Example not found")
        sys.exit(1)

    # Init other parameters
    np.random.seed(1294404794)
    np.seterr(all='ignore')
    best_L = np.inf                 # Best length (minimum)
    best_run = 0                    # Run corresponding to the best length
    best_count = 0                  # count corresponding to the best length
    paths = [None] * nRun           # List with the results from all runs
    result_text = [None] * nFrames     # position states (text)
    print("\nns = ", ns)

    # Set moving obstacles, start and goal in each frame
    for frame in range(nFrames):
        # Move the start position along the tangential direction of the
        # current path with a"speed" of 0.1
        # vel_s = 0.1
        # theta_s = np.arctan2(Py[1]-Py[0], Px[1]-Px[0])
        # x_s = layout.start[0] + vel_s * np.cos(theta_s)
        # y_s = layout.start[1] + vel_s * np.sin(theta_s)
        # fixed start
        x_s = layout.start[0]
        y_s = layout.start[1]
        z_s = layout.start[2]
        layout.set_start(x_s, y_s, z_s)

        # Move the goal position along a straight line with a "speed" of 0.15
        vel_g = 0.0     # fixed goal : set the speed to 0
        theta_g = -165.0 * np.pi / 180.0
        x_g = layout.goal[0] + vel_g * np.cos(theta_g)
        y_g = layout.goal[1] + vel_g * np.sin(theta_g)
        z_g = layout.goal[2]
        layout.set_goal(x_g, y_g, z_g)

        vel_o1 = 0.2
        vel_o2 = 0.3

        if (frame > 25):
            vel_o1 = -vel_o1  # Invert direction after 25 runs
        if frame > 45:
            vel_o2 = -vel_o2
        alpha_o = -60 * np.pi / 180.0
        theta_o = +150.0 * np.pi / 180.0
        x_o1 = layout.obs.get('Ball').get(frame)[0][0] + vel_o1 * np.cos(alpha_o) * np.cos(theta_o)
        y_o1 = layout.obs.get('Ball').get(frame)[0][1] + vel_o1 * np.cos(alpha_o) * np.sin(theta_o)
        z_o1 = layout.obs.get('Ball').get(frame)[0][2] + vel_o1 * np.sin(alpha_o)
        # layout.remove_obs('Ball', 0)
        layout.add_ball(frame+1,0, x_o1, y_o1, z_o1, r=0.75, Kv=100)

        x_o2 = layout.obs.get('Ball').get(frame)[1][0] + vel_o2 * np.cos(alpha_o) * np.cos(theta_o)
        y_o2 = layout.obs.get('Ball').get(frame)[1][1] + vel_o2 * np.cos(alpha_o) * np.sin(theta_o)
        z_o2 = layout.obs.get('Ball').get(frame)[1][2] + vel_o2 * np.sin(alpha_o)
        # layout.remove_obs('Ball', 1)
        layout.add_ball(frame+1,1, x_o2, y_o2, z_o2, r=0.75, Kv=100)


    # Run cases
    for run in range(nRun):
        # Optimize (the other PSO parameters have always their default values)
        layout.optimize(nCtrls=nCtrls, ns=ns, nPop=nPop, epochs=epochs,
                        f_interp=f_interp, Xinit=Xinit)

        # Save run
        paths[run] = deepcopy(layout)

        # Print results
        L = layout.sol[1]                   # Length
        col_count = layout.sol[2]           # Number of violated obstacles
        infeasible_count = layout.sol[3]    # Number of violated controls in planning frames
        print("\nrun={0:d}, L={1:.2f}, collision_count={2:d}, infeasibility_count={3:d}"
              .format(run+1, L, col_count, infeasible_count), end='', flush=True)

        # Save if best result (regardless the violations)
        if (L < best_L):
            best_L = L
            best_run = run
            best_count = col_count

            # Initialize the solution with the last optimal path
            Xinit = layout.sol[0]

        else:
            Xinit = None


    # Animation for example
    if (example == 'dynamic_ex'):
        tf = time.time()
        print("\nPSO solving time: ", tf - t0)

        def animate(row):

            # Path x and y coordinates
            path_sample.set_xdata(Px[row])
            path_sample.set_ydata(Py[row])
            # path.set_zdata(Pz_set[row])
            path_sample.set_3d_properties(Pz[row])

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

            # # Text with the path length
            L_text.set_text(result_text[row])
            # L_text.set_text(f'Path length: {lengths[row]:.2f}')

            # Obstacle and its centroid
            obs1.update(ax, new_center=(XYZc1[row, 0], XYZc1[row, 1], XYZc1[row, 2]))

            obs2.update(ax, new_center=(XYZc2[row, 0], XYZc2[row, 1], XYZc2[row, 2]))

            # return path, Ps, Pg, L_text, XYZobs1, obs1.surface

        start = np.zeros((nFrames, 3))  # Start position
        goal = np.zeros((nFrames, 3))   # Goal position
        XYZc1 = np.zeros((nFrames, 3))  # Obstacle1 (ball) centroid
        XYZc2 = np.zeros((nFrames, 3))  # Obstacle2 (ball) centroid

        layout = paths[-1]  # get the final optimized path
        Px = layout.sol[4].flatten()  # Path x coordinates in column
        Py = layout.sol[5].flatten()  # Path y coordinates in column
        Pz = layout.sol[6].flatten()  # Path z coordinates in column


        # Build the arrays needed for the animation
        for frame in range(nFrames):
            # Elements to be animated
            start[frame, :] = layout.start
            goal[frame, :] = layout.goal

            XYZc1[frame, :] = layout.obs.get('Ball').get(frame)[0][0:3]
            XYZc2[frame, :] = layout.obs.get('Ball').get(frame)[1][0:3]

            result_text[frame] = "run=" + str(run + 1) \
                            + ", L=" + str("{:.2f}".format(layout.sol[1])) \
                            + ", frame=" + str(frame)

        # Plot obstacles (except circle)
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111, aspect='equal', projection="3d")    # Create a 3D plot
        # paths[-1].remove_obs('Ball', 0)
        # paths[-1].remove_obs('Ball', 1)

        # Plot only what is inside the limits
        ax.set_xlim(paths[-1].limits[0], paths[-1].limits[1])
        ax.set_ylim(paths[-1].limits[2], paths[-1].limits[3])
        ax.set_zlim(paths[-1].limits[4], paths[-1].limits[5])
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        # Path
        path_sample = ax.plot(Px[0], Py[0], Pz[0], 'o', ms=4, c='b')[0]
        path = ax.plot(Px, Py, Pz, lw=0.5, c='b')[0]

        # Start and goal
        Ps = ax.plot(start[0, 0], start[0, 1], start[0, 2], 'o', ms=6, c='k')[0]
        Pg = ax.plot(goal[0, 0], goal[0, 1], goal[0, 2], '*', ms=8, c='k')[0]

        # Length (text)
        xt = paths[-1].limits[0] + 0.05 * (paths[-1].limits[1] - paths[-1].limits[0])
        yt = paths[-1].limits[2] + 0.05 * (paths[-1].limits[3] - paths[-1].limits[2])
        zt = paths[-1].limits[4] + 0.05 * (paths[-1].limits[5] - paths[-1].limits[4])
        L_text = ax.text(xt, yt, zt, result_text[0], fontsize=10)

        # Obstacle (ball)
        r = 0.75
        obs1 = Ball(ball_center=(XYZc1[0, 0], XYZc1[0, 1], XYZc1[0, 2]), radius=r, ball_color='wheat')
        obs1.plot(ax)
        # XYZobs1 = ax.plot(XYZc1[0, 0], XYZc1[0, 1], XYZc1[0, 2], 'x', ms=4, c='orange')[0]

        obs2 = Ball(ball_center=(XYZc2[0, 0], XYZc2[0, 1], XYZc2[0, 2]), radius=r, ball_color='wheat')
        obs2.plot(ax)
        # XYZobs2 = ax.plot(XYZc2[0, 0], XYZc2[0, 1], XYZc2[0, 2], 'x', ms=4, c='orange')[0]

        # Animate and save a copy
        anim = FuncAnimation(fig, animate, interval=200, frames=nFrames-1, blit=False)
        print("\n")
        # anim.save("Results_Example.gif")
        plt.show()


if __name__ == '__main__':
    main()