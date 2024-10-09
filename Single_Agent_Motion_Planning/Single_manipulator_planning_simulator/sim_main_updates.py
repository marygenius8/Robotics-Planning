import pybullet as p
import pybullet_data
import math
import util
from util import move_to_joint_pos, gripper_open, gripper_close
import yaml, json
from manipulator import Manipulator


class Config:
    """
    upaates this class data field with config.yaml properties.

    """
    def __int__(self, conf_file):
        self.robot =  conf_file

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

    manipulator = Manipulator(config)  # Initialize the manipulator with the config
    manipulator.simulate()  # Start the simulation


if __name__ == '__main__':
    main()
