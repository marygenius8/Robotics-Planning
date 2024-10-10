import pybullet as p
import pybullet_data
import math
import util
from util import move_to_joint_pos, gripper_open, gripper_close
import yaml, json
from manipulator_sim import ManipulatorSim


class Config:
    """
    upaates this class data field with config.yaml/json properties.

    """
    def __int__(self, cfg):
        self.robot_file = cfg['robot_urdf']

class ConfigLoader:
    def __init__(self, file_path, file_type='yaml'):
        self.file_path = file_path
        self.file_type = file_type
        self.config = None
        self.load_config()

    def load_config(self):
        """Reads the configuration from a YAML file."""
        if self.file_type == 'yaml':
            with open(self.file_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif self.file_type == 'json':
            with open(self.file_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError("Unsupported file type. Only YAML is supported for now.")

    def get_config(self):
        """Returns the loaded configuration."""
        return self.config


def main():
    config_loader = ConfigLoader('envConfig.json')  # Create an instance of the config loader
    config = Config(config_loader.get_config())  # Retrieve the configuration

    # Initialize PyBullet Simulation
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # Load PyBullet data

    target = p.getDebugVisualizerCamera()[11]
    p.resetDebugVisualizerCamera(
        cameraDistance=1.1,
        cameraYaw=90,
        cameraPitch=-25,
        cameraTargetPosition=[target[0], target[1], 0.7])

    p.setGravity(0, 0, -9.81)
    timeStep = 1. / 240.  # 240.
    p.setTimeStep(timeStep)

    manipulator = ManipulatorSim(config, p)  # Initialize the manipulator with the config
    # Plan a smooth trajectory
    trajectory = manipulator.plan_trajectory(config.start_conf, config.target_conf)

    # Run the simulation and move the manipulator along the trajectory
    manipulator.run_simulation(trajectory)

    # Disconnect PyBullet after simulation
    p.disconnect()


if __name__ == '__main__':
    main()
