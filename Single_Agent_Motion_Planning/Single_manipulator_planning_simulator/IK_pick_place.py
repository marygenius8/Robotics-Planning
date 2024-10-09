import pybullet as p
import time
import math
import pybullet_data

# Connect to PyBullet
clid = p.connect(p.GUI)  # Or use p.DIRECT for non-graphical
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load the environment: Plane, Desk, Ball, and Box
p.loadURDF("plane.urdf", [0, 0, -0.3])

# Load Desk (static object)
desk_id = p.loadURDF("cube.urdf", [0.5, 0, -0.3], globalScaling=1.0, useFixedBase=True)

# Load Ball on the desk
ball_id = p.loadURDF("sphere2.urdf", [0.45, 0, 0.2], globalScaling=0.05)

# Load Box on the desk (target location for ball)
box_id = p.loadURDF("cube.urdf", [0.65, 0, 0], globalScaling=0.2, useFixedBase=True)

# Load the KUKA iiwa manipulator
kuka_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
kukaEndEffectorIndex = 6  # End effector joint index
numJoints = p.getNumJoints(kuka_id)

# Define Joint Damping and other IK parameters
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]  # lower limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]  # upper limits for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]  # joint ranges
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]  # rest position
jd = [0.1] * numJoints  # Joint damping coefficients

# Set gravity
p.setGravity(0, 0, -9.81)

# Reset the joint state to the rest position
for i in range(numJoints):
    p.resetJointState(kuka_id, i, rp[i])


# Inverse Kinematics (IK) Helper Function
def move_to_position(position, orientation=[0, -math.pi, 0]):
    # Calculate joint positions using IK
    joint_poses = p.calculateInverseKinematics(kuka_id, kukaEndEffectorIndex, position,
                                               p.getQuaternionFromEuler(orientation), ll, ul, jr, rp)

    # Apply the joint positions to the robot
    for j in range(numJoints):
        p.setJointMotorControl2(kuka_id, j, p.POSITION_CONTROL, joint_poses[j])


# Simulation Loop for Pick-and-Place Task
p.setRealTimeSimulation(0)  # Use real-time simulation for smooth movement

# Step 1: Move to the pre-grasp position above the ball
pre_grasp_position = [0.45, 0, 0.3]  # Position above the ball
move_to_position(pre_grasp_position)
for _ in range(100):  # Simulate for a while
    p.stepSimulation()
    time.sleep(0.01)

# Step 2: Lower to the grasp position (near the ball)
grasp_position = [0.45, 0, 0.2]  # Position to grasp the ball
move_to_position(grasp_position)
for _ in range(100):  # Simulate for a while
    p.stepSimulation()
    time.sleep(0.01)

# Simulate the grasp action (no real gripper, but let's simulate picking the ball)
# Use constraints or teleport the ball to the end-effector
ball_constraint = p.createConstraint(kuka_id, kukaEndEffectorIndex, ball_id, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                     [0, 0, 0])

# Step 3: Move up with the ball (lifting the ball)
lift_position = [0.45, 0, 0.3]
move_to_position(lift_position)
for _ in range(100):  # Simulate for a while
    p.stepSimulation()
    time.sleep(0.01)

# Step 4: Move to the target box position above the box
pre_place_position = [0.65, 0, 0.3]  # Position above the box
move_to_position(pre_place_position)
for _ in range(100):
    p.stepSimulation()
    time.sleep(0.01)

# Step 5: Lower the ball into the box
place_position = [0.65, 0, 0.2]  # Position inside the box
move_to_position(place_position)
for _ in range(100):
    p.stepSimulation()
    time.sleep(0.01)

# Release the ball by removing the constraint
p.removeConstraint(ball_constraint)

# Step 6: Move back to a safe position (post-placement)
safe_position = [0.5, 0, 0.4]
move_to_position(safe_position)
for _ in range(100):
    p.stepSimulation()
    time.sleep(0.01)

# Final Disconnect
p.disconnect()
