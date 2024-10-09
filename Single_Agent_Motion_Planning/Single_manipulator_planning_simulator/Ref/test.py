import pybullet as p
import time
import math
import pybullet_data
from datetime import datetime

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0,0,0)

planeId = p.loadURDF("plane.urdf", [0, 0, -0.3])
startPos = [0, 0, 0]
# startOrientation = p.getQuaternionFromEuler([0,0,0])
kukaId = p.loadURDF("kuka_iiwa/model.urdf", startPos)
p.resetBasePositionAndOrientation(kukaId, [0, 0, 0], [0, 0, 0, 1])
kukaEndEffectorIndex = 6
numJoints = p.getNumJoints(kukaId)
if (numJoints != 7):
  exit()
#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(kukaId, startPos, startOrientation)

#lower limits for null space
ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
#upper limits for null space
ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
#joint ranges for null space
jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
#restposes for null space
rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
#joint damping coefficents
jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

t = 0.
prevPose = [0, 0, 0]
prevPose1 = [0, 0, 0]
hasPrevPose = 0
useNullSpace = 1

useOrientation = 1
#If we set useSimulation=0, it sets the arm pose to be the IK result directly without using dynamic control.
#This can be used to test the IK result accuracy.
useSimulation = 1
# useSimulation = 0
useRealTimeSimulation = 0
ikSolver = 0
p.setRealTimeSimulation(useRealTimeSimulation)
#trailDuration is duration (in seconds) after debug lines will be removed automatically
#use 0 for no-removal
trailDuration = 1
# Initial position (starting position)
initial_pos = [-0.4, 0.2, 0]
initial_orn = p.getQuaternionFromEuler([0, -math.pi, 0])

# Tolerance for stopping condition
tolerance = 0.01

while True:
    t += 0.01

    if useSimulation and not useRealTimeSimulation:
        p.stepSimulation()
    else:
        dt = datetime.now()
        t = (dt.second / 60.) * 2. * math.pi

    pos = [-0.4, 0.2 * math.cos(t), 0.2 * math.sin(t)]  # Circular trajectory
    orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # End effector orientation

    if useNullSpace:
        if useOrientation:
            jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, ll, ul, jr, rp)
        else:
            jointPoses = p.calculateInverseKinematics(kukaId,kukaEndEffectorIndex, pos, lowerLimits=ll, upperLimits=ul, jointRanges=jr, restPoses=rp)
    else:
        if useOrientation:
            jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, orn, jointDamping=jd, solver=ikSolver, maxNumIterations=100, residualThreshold=.01)
        else:
            jointPoses = p.calculateInverseKinematics(kukaId, kukaEndEffectorIndex, pos, solver=ikSolver)


    # Apply the joint positions
    if useSimulation:
        for i in range(numJoints):
            p.setJointMotorControl2(bodyIndex=kukaId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[i], targetVelocity=0, force=500,
                                    positionGain=0.03, velocityGain=1)
    else:
        # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(numJoints):
            p.resetJointState(kukaId, i, jointPoses[i])

    ls = p.getLinkState(kukaId, kukaEndEffectorIndex)

    if hasPrevPose:
        p.addUserDebugLine(prevPose, pos, [0, 0, 0.3], trailDuration)
        p.addUserDebugLine(prevPose1, ls[4], [1, 0, 0], trailDuration)

    prevPose = pos
    prevPose1 = ls[4]
    hasPrevPose = 1

    # # Check if the end effector is close to the initial position
    # if (abs(pos[0] - initial_pos[0]) < tolerance and
    #     abs(pos[1] - initial_pos[1]) < tolerance and
    #     abs(pos[2] - initial_pos[2]) < tolerance):
    #     print("Reached starting position. Stopping simulation.")
    #     break
    if t > 10:
        break

p.disconnect()
