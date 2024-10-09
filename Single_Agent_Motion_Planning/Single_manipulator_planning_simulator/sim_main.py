import pybullet as p
import pybullet_data
import math
import util
from util import move_to_joint_pos, gripper_open, gripper_close


def move_to_ee_pose(robot_id, target_ee_pos, target_ee_orientation=None):
    """
    Moves the robot to a given end-effector pose.
    :param robot_id: pyBullet's body id of the robot
    :param target_ee_pos: (3,) list/ndarray with target end-effector position
    :param target_ee_orientation: (4,) list/ndarray with target end-effector orientation as quaternion
    """
    # TODO (student): implement this function
    pass


def main():
    # connect to pybullet with a graphical user interface
    p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.7, 60, -30, [0.2, 0.2, 0.25])

    # basic configuration
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())  # allows us to load plane, robots, etc.
    plane_id = p.loadURDF('plane.urdf')  # function returns an ID for the loaded body

    # load the robot
    robot_id = p.loadURDF('franka_panda/panda.urdf', useFixedBase=True)

    # load an object to grasp and a box
    object_id = p.loadURDF('cube_small.urdf', basePosition=[0.5, -0.3, 0.025], baseOrientation=[0, 0, 0, 1])
    p.resetVisualShapeData(object_id, -1, rgbaColor=[1, 0, 0, 1])
    tray_id = p.loadURDF('tray/traybox.urdf', basePosition=[0.5, 0.5, 0.0], baseOrientation=[0, 0, 0, 1])

    # Define Joint Damping and other IK parameters
    numJoints = 7
    ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]  # lower limits for null space
    ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]  # upper limits for null space
    jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]  # joint ranges
    rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]  # rest position
    jd = [0.1] * numJoints  # Joint damping coefficients

    print('******************************')
    target_ori = p.getQuaternionFromEuler([0, -math.pi, 0])  # Pointing downwards

    init_ee_pos, quat, *_ = p.getLinkState(
        robot_id,
        util.ROBOT_EE_LINK_ID,
        computeForwardKinematics=True
    )
    move_ee_pos = list(init_ee_pos)[:]
    # move_ee_pos[0] = 0.5
    move_ee_pos[2] = move_ee_pos[2]-0.5

    config_init_ee = p.calculateInverseKinematics(
        robot_id,
        util.ROBOT_EE_LINK_ID,
        targetPosition=move_ee_pos,
        targetOrientation=target_ori,
        maxNumIterations=100,
        residualThreshold=0.001
    )

    pos_obj, ori_obj = p.getBasePositionAndOrientation(object_id)
    pos_obj_cube = list(pos_obj)
    config_obj_cube = p.calculateInverseKinematics(
        robot_id,
        util.ROBOT_EE_LINK_ID,
        targetPosition=pos_obj_cube,
        targetOrientation=target_ori,
        maxNumIterations=100,
        residualThreshold=0.001
    )
    print('config_cube:', config_obj_cube)
    pos_obj_cube_above = pos_obj_cube[:]
    pos_obj_cube_above[2] = pos_obj_cube_above[2] + 1.5

    # Use inverse kinematics to get the joint angles for the desired end-effector pose
    config_obj_cube_above = p.calculateInverseKinematics(
        robot_id,
        util.ROBOT_EE_LINK_ID,
        targetPosition=pos_obj_cube_above,
        targetOrientation=target_ori,
        maxNumIterations=100, lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses = rp,
        residualThreshold=0.01
    )
    print('config_cube_above:', config_obj_cube_above)

    pos_tray, ori_tray = p.getBasePositionAndOrientation(tray_id)
    pos_tray = list(pos_tray)
    pos_tray_above = pos_tray[:]
    pos_tray_above[2] = pos_tray_above[2] + 1
    config_tray_above = p.calculateInverseKinematics(
        robot_id,
        util.ROBOT_EE_LINK_ID,
        targetPosition=pos_tray_above,
        targetOrientation=target_ori,
        maxNumIterations=100,
        residualThreshold=0.001
    )
    print('config_tray_above:', config_tray_above)

    print('going to home configuration')
    move_to_joint_pos(robot_id, util.ROBOT_HOME_CONFIG)
    print('going to home configuration 0')
    move_to_joint_pos(robot_id, list(config_init_ee)[0:7])
    print('open gripper')
    gripper_open(robot_id)
    # print('moving to cube head above configuration 1')
    # move_to_joint_pos(robot_id, list(config_obj_cube_above)[0:7])
    print('moving to cube configuration 2')
    move_to_joint_pos(robot_id, list(config_obj_cube)[0:7])
    print('close gripper')
    gripper_close(robot_id)
    print('moving to tray head above configuration 3')
    move_to_joint_pos(robot_id, list(config_tray_above)[0:7])
    print('open gripper')
    gripper_open(robot_id)
    print('going to home configuration')
    move_to_joint_pos(robot_id, util.ROBOT_HOME_CONFIG)

    # clean up
    p.disconnect()


if __name__ == '__main__':
    main()
