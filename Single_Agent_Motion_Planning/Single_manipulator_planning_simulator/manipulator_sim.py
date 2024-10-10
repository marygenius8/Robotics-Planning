import pybullet as p
import pybullet_data
import time

class ManipulatorSim:
    def __init__(self, config, bullet, num_joints=6):
        """Initialize the pybullet simulation environment using configuration parameters."""
        self.config = config
        self.robot = bullet.loadURDF(config.robot_file, config.robot_start_pos, config.robot_start_orientation)
        self.num_joints = num_joints
        self.joint_limits = self.get_joint_limits()


    def get_joint_limits(self):
        # Define joint limits, which could be extracted from URDF or manually set
        joint_limits = {
            'lower': [-np.pi/2] * self.num_joints,
            'upper': [np.pi/2] * self.num_joints
        }
        return joint_limits


    def inverse_kinematics(self, target_pos, target_orientation):
        # Simple inverse kinematics
        joint_angles = p.calculateInverseKinematics(self.robot, self.config.end_effector_index, target_pos, target_orientation)
        return joint_angles



    def plan_trajectory(self, start_conf, goal_conf, num_steps=100):
        """
        trajectory planning method.
        """
        trajectory = []
        # Perform trajectory planning using the start_conf and goal_conf
        # the rrt star/mopso/chomp will be called here
        return trajectory

    def apply_joint_positions(self, joint_positions):
        """
        Set the joint positions of the manipulator in PyBullet.
        """
        for i in range(self.num_joints):
            p.setJointMotorControl2(self.robot, i, p.POSITION_CONTROL, targetPosition=joint_positions[i])


    def run_simulation(self, trajectory):
        """
        Run the PyBullet simulation step-by-step, following the planned trajectory.
        """
        for joint_positions in trajectory:
            self.apply_joint_positions(joint_positions)
            if self.check_collision():
                print("Collision detected, stopping simulation.")
                break
            p.stepSimulation()
            time.sleep(1. / 240.)  # Control the simulation step time
    
    def check_collision(self, other_object_id):
        collisions = p.getClosestPoints(self.robot_id, other_object_id, distance=0.01)
        return bool(collisions)