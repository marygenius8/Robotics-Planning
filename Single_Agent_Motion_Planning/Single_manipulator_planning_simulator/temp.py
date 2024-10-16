import numpy as np
import pybullet as p
import pybullet_data
import time
class Manipulator:
    def __init__(self, urdf_file, num_joints=6):
        self.urdf_file = urdf_file
        self.num_joints = num_joints
        self.robot_id = None
        self.joint_limits = self.get_joint_limits()

    def load_robot(self, start_pos=[0, 0, 0], start_orientation=[0, 0, 0, 1]):
        self.robot_id = p.loadURDF(self.urdf_file, start_pos, start_orientation)

    def inverse_kinematics(self, target_pos, target_orientation):
        # Simple inverse kinematics
        joint_angles = p.calculateInverseKinematics(self.robot_id, end_effector_index, target_pos, target_orientation)
        return joint_angles

    def plan_smooth_trajectory(self, start_conf, target_conf, num_steps=100):
        # Smooth trajectory planning
        t = np.linspace(0, 1, num_steps)
        cs = CubicSpline([0, 1], np.array([start_conf, target_conf]), axis=0)
        trajectory = cs(t)
        return trajectory

    def run_simulation(self, trajectory):
        for joint_positions in trajectory:
            self.apply_joint_positions(joint_positions)
            if self.check_collision():
                print("Collision detected, stopping simulation.")
                break
            p.stepSimulation()
            time.sleep(1. / 240.)

    def check_collision(self):
        collisions = p.getClosestPoints(self.robot_id, other_object_id, distance=0.01)
        return bool(collisions)

class Manipulator:
    def __init__(self, urdf_file, num_joints=6):
        self.urdf_file = urdf_file
        self.num_joints = num_joints
        self.robot_id = None
        self.joint_limits = self.get_joint_limits()

    def get_joint_limits(self):
        # Define joint limits, which could be extracted from URDF or manually set
        joint_limits = {
            'lower': [-np.pi/2] * self.num_joints,
            'upper': [np.pi/2] * self.num_joints
        }
        return joint_limits

    def load_robot(self, start_pos=[0, 0, 0], start_orientation=[0, 0, 0, 1]):
        # Load the URDF file of the manipulator into PyBullet
        self.robot_id = p.loadURDF(self.urdf_file, start_pos, start_orientation)

    def plan_trajectory(self, start_conf, target_conf, num_steps=100):
        """
        Trajectory planning method using linear interpolation between start and target configurations.
        """
        trajectory = []
        for t in np.linspace(0, 1, num_steps):
            step_conf = (1 - t) * np.array(start_conf) + t * np.array(target_conf)
            trajectory.append(step_conf)
        return trajectory

    def apply_joint_positions(self, joint_positions):
        """
        Set the joint positions of the manipulator in PyBullet.
        """
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])

    def run_simulation(self, trajectory):
        """
        Run the PyBullet simulation step-by-step, following the planned trajectory.
        """
        for joint_positions in trajectory:
            self.apply_joint_positions(joint_positions)
            p.stepSimulation()
            time.sleep(1./240.)  # Control the simulation step time

def main():


    # Create a manipulator object
    manipulator = Manipulator(urdf_file="kuka_iiwa/model.urdf")
    manipulator.load_robot()

    # Initial and target joint configurations for trajectory planning
    start_conf = [0, -0.5, 0.5, -1, 0.5, 0.5]
    target_conf = [0, 0.5, -0.5, 1, -0.5, -0.5]

    # Plan a smooth trajectory
    trajectory = manipulator.plan_trajectory(start_conf, target_conf)

    # Run the simulation and move the manipulator along the trajectory
    manipulator.run_simulation(trajectory)

    # Disconnect PyBullet after simulation
    p.disconnect()

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Function to update the animation frame by frame
def update(frame):
    q = path[frame]
    x_points, y_points, z_points = seven_joint_arm.get_points(q)
    ax = rrt_star.draw_graph()
    # Plot current configuration in grey
    for i, q_temp in enumerate(path[:frame + 1]):
        x_temp, y_temp, z_temp = seven_joint_arm.get_points(q_temp)
        ax.plot([x for x in x_temp],
                [y for y in y_temp],
                [z for z in z_temp],
                "o-", color="grey", ms=4, mew=0.5)

    # Plot final configuration in red
    x_final, y_final, z_final = seven_joint_arm.get_points(path[-1])
    ax.plot([x for x in x_final],
            [y for y in y_final],
            [z for z in z_final],
            "o-", color="red", ms=5, mew=0.5)

# Initialize plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
rrt_star.draw_graph(ax)  # Initial graph

# Create the animation
ani = FuncAnimation(fig, update, frames=len(path), repeat=False)

# Save the animation as a GIF
ani.save('robot_arm_motion.gif', writer='pillow', fps=10)

plt.show()
