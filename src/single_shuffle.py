# ----------------------------------------------------------------------------------------------------------------------
# Licensing Information: You are free to use or extend these projects for
# education or reserach purposes provided that you provide clear attribution to UC Berkeley,
# including a reference to the papers describing the control framework:
# [1] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks. A Data-Driven
#     Control Framework." In IEEE Transactions on Automatic Control (2017).
#
# [2] Ugo Rosolia, Ashwin Carvalho, and Francesco Borrelli. "Autonomous racing using learning model predictive control."
#     In 2017 IEEE American Control Conference (ACC)
#
# [3] Maximilian Brunner, Ugo Rosolia, Jon Gonzales and Francesco Borrelli "Repetitive learning model predictive
#     control: An autonomous racing example" In 2017 IEEE Conference on Decision and Control (CDC)
#
# [4] Ugo Rosolia and Francesco Borrelli. "Learning Model Predictive Control for Iterative Tasks: A Computationally
#     Efficient Approach for Linear System." IFAC-PapersOnLine 50.1 (2017).
#
# Attibution Information: Code developed by Ugo Rosolia
# (for clarifications and suggestions please write to ugo.rosolia@berkeley.edu).
#
# Code description: Simulation of the Learning Model Predictive Controller (LMPC). The main file runs:
# 1) A PID path following controller
# 2) A Model Predictive Controller (MPC) which uses a LTI model identified from the data collected with the PID in 1)
# 3) A MPC which uses a LTV model identified from the date collected in 1)
# 4) A LMPC for racing where the safe set and value function approximation are build using the data from 1), 2) and 3)
# ----------------------------------------------------------------------------------------------------------------------

import sys
sys.path.append('fnc')
from SysModel import Simulator, PID
from Classes import ClosedLoopData, LMPCprediction
from PathFollowingLTVMPC import PathFollowingLTV_MPC
from PathFollowingLTIMPC import PathFollowingLTI_MPC
from Track import Map, unityTestChangeOfCoordinates
from LMPC import ControllerLMPC, PWAControllerLMPC
from Utilities import Regression
from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy, animation_states, saveGif_xyResults, Save_statesAnimation, plotMap, plotSafeSet
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle

# ======================================================================================================================
# ============================ Choose which controller to run to set up problem ========================================
# ======================================================================================================================
RunPID     = 1; plotFlag       = 0
RunMPC     = 1; plotFlagMPC    = 0
RunMPC_tv  = 1; plotFlagMPC_tv = 0
RunLMPC    = 1; plotFlagLMPC   = 1; animation_xyFlag = 1; animation_stateFlag = 0
runPWAFlag = 0; # uncomment importing pwa_cluster in LMPC.py
testCoordChangeFlag = 0;
plotOneStepPredictionErrors = 1;

# ======================================================================================================================
# ============================ Initialize parameters for path following ================================================
# ======================================================================================================================
dt         = 1.0/10.0        # Controller discretization time
Time       = 100             # Simulation time for path following PID
TimeMPC    = 100             # Simulation time for path following MPC
TimeMPC_tv = 100             # Simulation time for path following LTV-MPC
vt         = 0.8             # Reference velocity for path following controllers
v0         = 0.5             # Initial velocity at lap 0
N          = 12            # Horizon length
n = 6;   d = 2               # State and Input dimension

# Path Following tuning
Q = np.diag([1.0, 1.0, 1, 1, 0.0, 100.0]) # vx, vy, wz, epsi, s, ey
R = np.diag([1.0, 10.0])                  # delta, a

map = Map(0.8)                            # Initialize the map (PointAndTangent); argument is track width
simulator = Simulator(map)                # Initialize the Simulator

# ======================================================================================================================
# ==================================== Initialize parameters for LMPC ==================================================
# ======================================================================================================================
TimeLMPC   = 400              # Simulation time
Laps       = 5+2              # Total LMPC laps

# Safe Set Parameter
LMPC_Solver = "CVX"           # Can pick CVX for cvxopt or OSQP. For OSQP uncomment line 14 in LMPC.py
numSS_it = 2                  # Number of trajectories used at each iteration to build the safe set
numSS_Points = 32 + N         # Number of points to select from each trajectory to build the safe set
shift = 0                     # Given the closed point, x_t^j, to the x(t) select the SS points from x_{t+shift}^j

# Tuning Parameters
Qslack  = 5*np.diag([10, 1, 1, 1, 10, 1])          # Cost on the slack variable for the terminal constraint
Q_LMPC  =  0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # State cost x = [vx, vy, wz, epsi, s, ey]
R_LMPC  =  1 * np.diag([1.0, 1.0])                      # Input cost u = [delta, a]
dR_LMPC =  5 * np.array([1.0, 1.0])                     # Input rate cost u

# Initialize LMPC simulator
LMPCSimulator = Simulator(map, 1, 1) #flags indicate one lap, and using the LMPC controller

# ======================================================================================================================
# ======================================= PID path following ===========================================================
# ======================================================================================================================
print("Starting PID")
if RunPID == 1:
    ClosedLoopDataPID = ClosedLoopData(dt, Time , v0) #form matrices for experiment data
    PIDController = PID(vt) #sets the reference velocity and some timers?
    simulator.Sim(ClosedLoopDataPID, PIDController) #simulates the PID controller for Time timesteps

    file_data = open('data/ClosedLoopDataPID.obj', 'wb')
    pickle.dump(ClosedLoopDataPID, file_data)
    file_data.close()
else:
    file_data = open('data/ClosedLoopDataPID.obj', 'rb')
    ClosedLoopDataPID = pickle.load(file_data)
    file_data.close()
print("===== PID terminated")
if plotFlag == 1:
    plotTrajectory(map, ClosedLoopDataPID.x, ClosedLoopDataPID.x_glob, ClosedLoopDataPID.u)
    plt.show()
# ======================================================================================================================
# ======================================  LINEAR REGRESSION ============================================================
# ======================================================================================================================
raw_input("Finished PID - Start TI-MPC?")
print("Starting TI-MPC")
lamb = 0.0000001
#fit linear dynamics to the closed loop data: x2 = A*x1 + b*u1; lamb is weight on frob norm of W
A, B, Error = Regression(ClosedLoopDataPID.x, ClosedLoopDataPID.u, lamb)

if RunMPC == 1:
    ClosedLoopDataLTI_MPC = ClosedLoopData(dt, TimeMPC, v0) #form (empty) matrices for experiment data
    Controller_PathFollowingLTI_MPC = PathFollowingLTI_MPC(A, B, Q, R, N, vt)
    simulator.Sim(ClosedLoopDataLTI_MPC, Controller_PathFollowingLTI_MPC)

    #file_data = open(sys.path[0]+'/data/ClosedLoopDataLTI_MPC.obj', 'wb')
    file_data = open('data/ClosedLoopDataLTI_MPC.obj', 'wb')
    pickle.dump(ClosedLoopDataLTI_MPC, file_data)
    file_data.close()
else:
    #file_data = open(sys.path[0]+'/data/ClosedLoopDataLTI_MPC.obj', 'rb')
    file_data = open('data/ClosedLoopDataLTI_MPC.obj', 'rb')
    ClosedLoopDataLTI_MPC = pickle.load(file_data)
    file_data.close()
print("===== TI-MPC terminated")
if plotFlagMPC == 1:
    plotTrajectory(map, ClosedLoopDataLTI_MPC.x, ClosedLoopDataLTI_MPC.x_glob, ClosedLoopDataLTI_MPC.u)
    plt.show()
# ======================================================================================================================
# ===================================  LOCAL LINEAR REGRESSION =========================================================
# ======================================================================================================================
raw_input("Finished TI-MPC - Start TV-MPC?")
print("Starting TV-MPC")
if RunMPC_tv == 1:
    ClosedLoopDataLTV_MPC = ClosedLoopData(dt, TimeMPC_tv, v0)
    Controller_PathFollowingLTV_MPC = PathFollowingLTV_MPC(Q, R, N, vt, n, d, ClosedLoopDataPID.x, ClosedLoopDataPID.u, dt, map)
    simulator.Sim(ClosedLoopDataLTV_MPC, Controller_PathFollowingLTV_MPC)

    #file_data = open(sys.path[0]+'/data/ClosedLoopDataLTV_MPC.obj', 'wb')
    file_data = open('data/ClosedLoopDataLTV_MPC.obj', 'wb')
    pickle.dump(ClosedLoopDataLTV_MPC, file_data)
    file_data.close()
else:
    #file_data = open(sys.path[0]+'/data/ClosedLoopDataLTV_MPC.obj', 'rb')
    file_data = open('data/ClosedLoopDataLTV_MPC.obj', 'rb')
    ClosedLoopDataLTV_MPC = pickle.load(file_data)
    file_data.close()
print("===== TV-MPC terminated")
if plotFlagMPC_tv == 1:
    plotTrajectory(map, ClosedLoopDataLTV_MPC.x, ClosedLoopDataLTV_MPC.x_glob, ClosedLoopDataLTV_MPC.u)
    plt.show()

# ======================================================================================================================
# ==============================  LMPC w\ LOCAL LINEAR REGRESSION ======================================================
# ======================================================================================================================
raw_input("Finished TV-MPC - Start LMPC?")
print("Starting LMPC")
ClosedLoopLMPC = ClosedLoopData(dt, TimeLMPC, v0)
LMPCOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, Laps) #to store open-loop prediction and safe sets
LMPCSimulator = Simulator(map, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON

if runPWAFlag == 1:
    LMPController = PWAControllerLMPC(10, numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)
else:
    LMPController = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)
    onlyLMPController = ControllerLMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)

# add previously completed trajectories to Safe Set: 
LMPController.addTrajectory(ClosedLoopDataPID)
LMPController.addTrajectory(ClosedLoopDataLTV_MPC)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopLMPC.x[0,:]
x0_glob[0,:] = ClosedLoopLMPC.x_glob[0,:]

if RunLMPC == 1:
    for it in range(2, Laps):

        ClosedLoopLMPC.updateInitialConditions(x0, x0_glob)
        LMPCSimulator.Sim(ClosedLoopLMPC, LMPController, LMPCOpenLoopData) #this runs one lap at a time due to initialization!
        LMPController.addTrajectory(ClosedLoopLMPC)
        onlyLMPController.addTrajectory(ClosedLoopLMPC)

        if LMPController.feasible == 0:
            break
        else:
            # Reset Initial Conditions
            x0[0,:]      = ClosedLoopLMPC.x[ClosedLoopLMPC.SimTime, :] - np.array([0, 0, 0, 0, map.TrackLength, 0])
            x0_glob[0,:] = ClosedLoopLMPC.x_glob[ClosedLoopLMPC.SimTime, :]

    #file_data = open(sys.path[0]+'/data/LMPController.obj', 'wb')
    file_data = open('data/LMPController.obj', 'wb')
    pickle.dump(ClosedLoopLMPC, file_data)
    pickle.dump(LMPController, file_data)
    pickle.dump(LMPCOpenLoopData, file_data)
    file_data.close()
    
    file_data = open('data/onlyLMPController.obj','wb')
    pickle.dump(onlyLMPController,file_data)
    file_data.close()
else:
    #file_data = open(sys.path[0]+'/data/LMPController.obj', 'rb')
    file_data = open('data/LMPController.obj', 'rb')
    ClosedLoopLMPC = pickle.load(file_data)
    LMPController  = pickle.load(file_data)
    LMPCOpenLoopData  = pickle.load(file_data)
    file_data.close()
    
    file_data = open('data/onlyLMPController.obj','rb')
    onlyLMPController = pickle.load(file_data)

print("===== LMPC terminated")

if plotFlagLMPC == 1:
    plotClosedLoopLMPC(LMPController, map)
    plt.show()

# plot the safe set along the map 
plotSafeSet(onlyLMPController.SS,map)
raw_input("LMPC on original track is done.")

# <codecell>
# ======================================================================================================================
# ========================================= TRACK/SAFE SET RESHUFFLING =================================================
# ======================================================================================================================
# pre-process LMPC.Qfun to avoid the negatives
#onlyLMPController.processQfun()
onlyLMPController.processQfun()

# split safe set into modes
#onlyLMPController.splitTheSS(map)
onlyLMPController.splitTheSS(map)
    
# relativize safe set (set intial s --> 0)
#onlyLMPController.relTheSplitSS(map)
onlyLMPController.relTheSplitSS(map)
   
shuffledWell = False

while not shuffledWell: 
    # shuffle safe set according to new track
    shuffledMap = map.shuffle()    

    # turn relative safe set into absolute coordinates again (in modes)
    #onlyLMPController.makeShuffledSS(shuffledMap)
    onlyLMPController.makeShuffledSS(shuffledMap)
#plotSafeSet(LMPController.shuffledSS, shuffledMap)
#raw_input("Shuffling of original safe set is done.")

# ======================================================================================================================
# ========================================= REACHABILITY ANALYSIS ======================================================
# ======================================================================================================================
    #onlyLMPController.reachabilityAnalysis(A,B,Qslack,N)
    onlyLMPController.reachabilityAnalysis(A,B,Qslack,N)
    
    shuffledWell = bool(input("Shuffled correctly?"))   

plotSafeSet(onlyLMPController.reachableSS,shuffledMap)

#onlyLMPController.reorganizeReachableSafeSet()
onlyLMPController.reorganizeReachableSafeSet()
raw_input("Reachability analysis on new track is done.")


# <codecell> 
# ======================================================================================================================
# ========================================= LMPC on SHUFFLED TRACK =====================================================
# ======================================================================================================================
# We will compare performance of the LMPC controller on the shuffled track. In particular, we consider two cases:
#   1. Performance of the LMPC controller initialized with the PID+TVMPC Safe Set
#   2. Performance of the LMPC controller initialized with the Reachable+TVMPC SafeSet 
# Performance will be evaluated on iterations required to traverse the course

# ========================================= 0. Set up for simulation on new track ======================================
simulatorPID = Simulator(shuffledMap) 
simulatorPID.laps = 1

# ========================================= 0a. Run the PID controller on new track =====================================
ClosedLoopDataShuffledPID = ClosedLoopData(dt, 0.5*Time , v0) #form matrices for experiment data
ShuffledPIDController = PID(vt) #sets the reference velocity and some timers?
simulatorPID.Sim(ClosedLoopDataShuffledPID, ShuffledPIDController) #simulates the PID controller for Time timesteps
file_data = open('data/ClosedLoopDataShuffledPID.obj', 'wb')
pickle.dump(ClosedLoopDataShuffledPID, file_data)
file_data.close()

plotTrajectory(shuffledMap, ClosedLoopDataShuffledPID.x, ClosedLoopDataShuffledPID.x_glob, ClosedLoopDataShuffledPID.u)
plt.show()

# ========================================= 0b. Learn TI from PID data =================================================
#fit linear dynamics to the closed loop data: x2 = A*x1 + b*u1; lamb is weight on frob norm of W
#lamb = 0.0000001
#A, B, Error = Regression(ClosedLoopDataShuffledPID.x, ClosedLoopDataShuffledPID.u, lamb)
#ClosedLoopDataShuffledLTI_MPC = ClosedLoopData(dt, TimeMPC, v0) #form (empty) matrices for experiment data
#ShuffledController_PathFollowingLTI_MPC = PathFollowingLTI_MPC(A, B, Q, R, N, vt)
#simulatorPID.Sim(ClosedLoopDataShuffledLTI_MPC, ShuffledController_PathFollowingLTI_MPC)
#file_data = open('data/ClosedLoopDataShuffledLTI_MPC.obj', 'wb')
#pickle.dump(ClosedLoopDataShuffledLTI_MPC, file_data)
#file_data.close()
#
#plotTrajectory(shuffledMap, ClosedLoopDataShuffledLTI_MPC.x, ClosedLoopDataShuffledLTI_MPC.x_glob, ClosedLoopDataShuffledLTI_MPC.u)
#plt.show()
#
## ========================================= 0c. Learn TV from PID data =================================================
#ClosedLoopDataShuffledLTV_MPC = ClosedLoopData(dt, TimeMPC_tv, v0)
#ShuffledController_PathFollowingLTV_MPC = PathFollowingLTV_MPC(Q, R, N, vt, n, d, ClosedLoopDataShuffledPID.x, ClosedLoopDataShuffledPID.u, dt, map)
#simulatorPID.Sim(ClosedLoopDataShuffledLTV_MPC, ShuffledController_PathFollowingLTV_MPC)
#file_data = open('data/ClosedLoopDataShuffledLTV_MPC.obj', 'wb')
#pickle.dump(ClosedLoopDataShuffledLTV_MPC, file_data)
#file_data.close()
#
#plotTrajectory(shuffledMap, ClosedLoopDataShuffledLTV_MPC.x, ClosedLoopDataShuffledLTV_MPC.x_glob, ClosedLoopDataShuffledLTV_MPC.u)
#plt.show()

# <codecell> PID Experiment
raw_input("Going to start running PID-N Experiment")
ShuffledLaps = 20
N1 = 10
N2 = 14
N3 = 18

# This controller will start with a safe set consisting of PID and TVMPC laps
ClosedLoopShuffledLMPC11 = ClosedLoopData(dt, TimeLMPC, v0)
ShuffledLMPC11OpenLoopData = LMPCprediction(N1, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
ShuffledLMPC11Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
ShuffledLMPC11Controller = ControllerLMPC(numSS_Points, numSS_it, N1, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)

ShuffledLMPC11Controller.addTrajectory(ClosedLoopDataShuffledPID)
ShuffledLMPC11Controller.addTrajectory(ClosedLoopDataShuffledPID)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopShuffledLMPC11.x[0,:]
x0_glob[0,:] = ClosedLoopShuffledLMPC11.x_glob[0,:]

for it in range(2, ShuffledLaps):
    #ShuffledLMPC1Controller.numSS_it = ShuffledLMPC1Controller.it
    ClosedLoopShuffledLMPC11.updateInitialConditions(x0, x0_glob)
    ShuffledLMPC11Simulator.Sim(ClosedLoopShuffledLMPC11, ShuffledLMPC11Controller, ShuffledLMPC11OpenLoopData) #this runs one lap at a time due to initialization!
    ShuffledLMPC11Controller.addTrajectory(ClosedLoopShuffledLMPC11)

    if ShuffledLMPC11Controller.feasible == 0:
        break
    else:
        x0[0,:]      = ClosedLoopShuffledLMPC11.x[0,:]
        x0_glob[0,:] = ClosedLoopShuffledLMPC11.x_glob[0,:]
        #x0[0,:]      = ClosedLoopShuffledLMPC11.x[ClosedLoopShuffledLMPC11.SimTime, :] - np.array([0, 0, 0, 0, shuffledMap.TrackLength, 0])
        #x0_glob[0,:] = ClosedLoopShuffledLMPC11.x_glob[ClosedLoopShuffledLMPC11.SimTime, :]

################
raw_input("N2")
ClosedLoopShuffledLMPC12 = ClosedLoopData(dt, TimeLMPC, v0)
ShuffledLMPC12OpenLoopData = LMPCprediction(N2, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
ShuffledLMPC12Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
ShuffledLMPC12Controller = ControllerLMPC(numSS_Points, numSS_it, N2, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)

ShuffledLMPC12Controller.addTrajectory(ClosedLoopDataShuffledPID)
ShuffledLMPC12Controller.addTrajectory(ClosedLoopDataShuffledPID)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopShuffledLMPC12.x[0,:]
x0_glob[0,:] = ClosedLoopShuffledLMPC12.x_glob[0,:]

for it in range(2, ShuffledLaps):
    #ShuffledLMPC1Controller.numSS_it = ShuffledLMPC1Controller.it
    ClosedLoopShuffledLMPC12.updateInitialConditions(x0, x0_glob)
    ShuffledLMPC12Simulator.Sim(ClosedLoopShuffledLMPC12, ShuffledLMPC12Controller, ShuffledLMPC12OpenLoopData) #this runs one lap at a time due to initialization!
    ShuffledLMPC12Controller.addTrajectory(ClosedLoopShuffledLMPC12)

    if ShuffledLMPC12Controller.feasible == 0:
        break
    else:
        x0[0,:]      = ClosedLoopShuffledLMPC12.x[0,:]
        x0_glob[0,:] = ClosedLoopShuffledLMPC12.x_glob[0,:]
        #x0[0,:]      = ClosedLoopShuffledLMPC11.x[ClosedLoopShuffledLMPC12.SimTime, :] - np.array([0, 0, 0, 0, shuffledMap.TrackLength, 0])
        #x0_glob[0,:] = ClosedLoopShuffledLMPC11.x_glob[ClosedLoopShuffledLMPC12.SimTime, :]
        
################
raw_input("N3")
ClosedLoopShuffledLMPC13 = ClosedLoopData(dt, TimeLMPC, v0)
ShuffledLMPC13OpenLoopData = LMPCprediction(N3, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
ShuffledLMPC13Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
ShuffledLMPC13Controller = ControllerLMPC(numSS_Points, numSS_it, N3, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)

ShuffledLMPC13Controller.addTrajectory(ClosedLoopDataShuffledPID)
ShuffledLMPC13Controller.addTrajectory(ClosedLoopDataShuffledPID)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopShuffledLMPC13.x[0,:]
x0_glob[0,:] = ClosedLoopShuffledLMPC13.x_glob[0,:]

for it in range(2, ShuffledLaps):
    #ShuffledLMPC1Controller.numSS_it = ShuffledLMPC1Controller.it
    ClosedLoopShuffledLMPC13.updateInitialConditions(x0, x0_glob)
    ShuffledLMPC13Simulator.Sim(ClosedLoopShuffledLMPC13, ShuffledLMPC13Controller, ShuffledLMPC13OpenLoopData) #this runs one lap at a time due to initialization!
    ShuffledLMPC13Controller.addTrajectory(ClosedLoopShuffledLMPC13)

    if ShuffledLMPC13Controller.feasible == 0:
        break
    else:
        x0[0,:]      = ClosedLoopShuffledLMPC13.x[0,:]
        x0_glob[0,:] = ClosedLoopShuffledLMPC13.x_glob[0,:]
        #x0[0,:]      = ClosedLoopShuffledLMPC11.x[ClosedLoopShuffledLMPC13.SimTime, :] - np.array([0, 0, 0, 0, shuffledMap.TrackLength, 0])
        #x0_glob[0,:] = ClosedLoopShuffledLMPC11.x_glob[ClosedLoopShuffledLMPC13.SimTime, :]
        
################
#raw_input("N4")
#ClosedLoopShuffledLMPC14 = ClosedLoopData(dt, TimeLMPC, v0)
#ShuffledLMPC14OpenLoopData = LMPCprediction(N4, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
#ShuffledLMPC14Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
#ShuffledLMPC14Controller = ControllerLMPC(numSS_Points, numSS_it, N4, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)
#
#ShuffledLMPC14Controller.addTrajectory(ClosedLoopDataShuffledPID)
#ShuffledLMPC14Controller.addTrajectory(ClosedLoopDataShuffledPID)
#
#x0           = np.zeros((1,n))
#x0_glob      = np.zeros((1,n))
#x0[0,:]      = ClosedLoopShuffledLMPC14.x[0,:]
#x0_glob[0,:] = ClosedLoopShuffledLMPC14.x_glob[0,:]
#
#for it in range(2, ShuffledLaps):
#    #ShuffledLMPC1Controller.numSS_it = ShuffledLMPC1Controller.it
#    ClosedLoopShuffledLMPC14.updateInitialConditions(x0, x0_glob)
#    ShuffledLMPC14Simulator.Sim(ClosedLoopShuffledLMPC14, ShuffledLMPC14Controller, ShuffledLMPC14OpenLoopData) #this runs one lap at a time due to initialization!
#    ShuffledLMPC14Controller.addTrajectory(ClosedLoopShuffledLMPC14)
#
#    if ShuffledLMPC14Controller.feasible == 0:
#        break
#    else:
#        x0[0,:]      = ClosedLoopShuffledLMPC14.x[0,:]
#        x0_glob[0,:] = ClosedLoopShuffledLMPC14.x_glob[0,:]
#        #x0[0,:]      = ClosedLoopShuffledLMPC11.x[ClosedLoopShuffledLMPC14.SimTime, :] - np.array([0, 0, 0, 0, shuffledMap.TrackLength, 0])
#        #x0_glob[0,:] = ClosedLoopShuffledLMPC11.x_glob[ClosedLoopShuffledLMPC14.SimTime, :]

# <codecell> 
# ========================================= 2 Create and run LMPC controller 2 ========================================= 
raw_input("Going to start running Shuffled LMPC2")
# This controller will start with a safe set consisting of RSS and TVMPC laps

ShuffledLaps = 20

ClosedLoopShuffledLMPC21 = ClosedLoopData(dt, TimeLMPC, v0)
ShuffledLMPC21OpenLoopData = LMPCprediction(N1, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
ShuffledLMPC21Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
ShuffledLMPC21Controller = ControllerLMPC(numSS_Points, numSS_it, N1, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)

rSS, ruSS, rQfun = onlyLMPController.selectBestTrajectory()
ShuffledLMPC21Controller.addReachableSet(rSS,ruSS,rQfun,shuffledMap) #################### FIX THIS FIX THIS
ShuffledLMPC21Controller.addTrajectory(ClosedLoopDataShuffledPID)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopShuffledLMPC21.x[0,:]
x0_glob[0,:] = ClosedLoopShuffledLMPC21.x_glob[0,:]

for it in range(2, ShuffledLaps):
    #ShuffledLMPC2Controller.numSS_it = ShuffledLMPC2Controller.it
    ClosedLoopShuffledLMPC21.updateInitialConditions(x0, x0_glob)
    ShuffledLMPC21Simulator.Sim(ClosedLoopShuffledLMPC21, ShuffledLMPC21Controller, ShuffledLMPC21OpenLoopData) #this runs one lap at a time due to initialization!
    ShuffledLMPC21Controller.addTrajectory(ClosedLoopShuffledLMPC21)

    if ShuffledLMPC21Controller.feasible == 0:
        break
    else:
        # Reset Initial Conditions
        x0[0,:]      = ClosedLoopShuffledLMPC21.x[0,:]
        x0_glob[0,:] = ClosedLoopShuffledLMPC21.x_glob[0,:]
        
######################
ClosedLoopShuffledLMPC22 = ClosedLoopData(dt, TimeLMPC, v0)
ShuffledLMPC22OpenLoopData = LMPCprediction(N2, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
ShuffledLMPC22Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
ShuffledLMPC22Controller = ControllerLMPC(numSS_Points, numSS_it, N2, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)

rSS, ruSS, rQfun = onlyLMPController.selectBestTrajectory()
ShuffledLMPC22Controller.addReachableSet(rSS,ruSS,rQfun,shuffledMap) #################### FIX THIS FIX THIS
ShuffledLMPC22Controller.addTrajectory(ClosedLoopDataShuffledPID)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopShuffledLMPC22.x[0,:]
x0_glob[0,:] = ClosedLoopShuffledLMPC22.x_glob[0,:]

for it in range(2, ShuffledLaps):
    #ShuffledLMPC2Controller.numSS_it = ShuffledLMPC2Controller.it
    ClosedLoopShuffledLMPC22.updateInitialConditions(x0, x0_glob)
    ShuffledLMPC22Simulator.Sim(ClosedLoopShuffledLMPC22, ShuffledLMPC22Controller, ShuffledLMPC22OpenLoopData) #this runs one lap at a time due to initialization!
    ShuffledLMPC22Controller.addTrajectory(ClosedLoopShuffledLMPC22)

    if ShuffledLMPC22Controller.feasible == 0:
        break
    else:
        # Reset Initial Conditions
        x0[0,:]      = ClosedLoopShuffledLMPC22.x[0,:]
        x0_glob[0,:] = ClosedLoopShuffledLMPC22.x_glob[0,:]

######################
ClosedLoopShuffledLMPC23 = ClosedLoopData(dt, TimeLMPC, v0)
ShuffledLMPC23OpenLoopData = LMPCprediction(N3, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
ShuffledLMPC23Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
ShuffledLMPC23Controller = ControllerLMPC(numSS_Points, numSS_it, N3, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)

rSS, ruSS, rQfun = onlyLMPController.selectBestTrajectory()
ShuffledLMPC23Controller.addReachableSet(rSS,ruSS,rQfun,shuffledMap) #################### FIX THIS FIX THIS
ShuffledLMPC23Controller.addTrajectory(ClosedLoopDataShuffledPID)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopShuffledLMPC23.x[0,:]
x0_glob[0,:] = ClosedLoopShuffledLMPC23.x_glob[0,:]

for it in range(2, ShuffledLaps):
    #ShuffledLMPC2Controller.numSS_it = ShuffledLMPC2Controller.it
    ClosedLoopShuffledLMPC23.updateInitialConditions(x0, x0_glob)
    ShuffledLMPC23Simulator.Sim(ClosedLoopShuffledLMPC23, ShuffledLMPC23Controller, ShuffledLMPC23OpenLoopData) #this runs one lap at a time due to initialization!
    ShuffledLMPC23Controller.addTrajectory(ClosedLoopShuffledLMPC23)

    if ShuffledLMPC23Controller.feasible == 0:
        break
    else:
        # Reset Initial Conditions
        x0[0,:]      = ClosedLoopShuffledLMPC23.x[0,:]
        x0_glob[0,:] = ClosedLoopShuffledLMPC23.x_glob[0,:]
        
######################
#ClosedLoopShuffledLMPC24 = ClosedLoopData(dt, TimeLMPC, v0)
#ShuffledLMPC24OpenLoopData = LMPCprediction(N4, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
#ShuffledLMPC24Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
#ShuffledLMPC24Controller = ControllerLMPC(numSS_Points, numSS_it, N4, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)
#
#rSS, ruSS, rQfun = onlyLMPController.selectBestTrajectory()
#ShuffledLMPC24Controller.addReachableSet(rSS,ruSS,rQfun,shuffledMap) #################### FIX THIS FIX THIS
#ShuffledLMPC24Controller.addTrajectory(ClosedLoopDataShuffledPID)
#
#x0           = np.zeros((1,n))
#x0_glob      = np.zeros((1,n))
#x0[0,:]      = ClosedLoopShuffledLMPC24.x[0,:]
#x0_glob[0,:] = ClosedLoopShuffledLMPC24.x_glob[0,:]
#
#for it in range(2, ShuffledLaps):
#    #ShuffledLMPC2Controller.numSS_it = ShuffledLMPC2Controller.it
#    ClosedLoopShuffledLMPC24.updateInitialConditions(x0, x0_glob)
#    ShuffledLMPC24Simulator.Sim(ClosedLoopShuffledLMPC24, ShuffledLMPC24Controller, ShuffledLMPC24OpenLoopData) #this runs one lap at a time due to initialization!
#    ShuffledLMPC24Controller.addTrajectory(ClosedLoopShuffledLMPC24)
#
#    if ShuffledLMPC24Controller.feasible == 0:
#        break
#    else:
#        # Reset Initial Conditions
#        x0[0,:]      = ClosedLoopShuffledLMPC24.x[0,:]
#        x0_glob[0,:] = ClosedLoopShuffledLMPC24.x_glob[0,:]
#


plt.figure()
plt.plot(range(2,ShuffledLMPC11Controller.it), ShuffledLMPC11Controller.Qfun[0,2:]*dt,label='N1-initialized')
plt.plot(range(2,ShuffledLMPC12Controller.it), ShuffledLMPC12Controller.Qfun[0,2:]*dt,label='N2-initialized')
plt.plot(range(2,ShuffledLMPC13Controller.it), ShuffledLMPC13Controller.Qfun[0,2:]*dt,label='N3-initialized')
#plt.plot(range(2,ShuffledLMPC14Controller.it), ShuffledLMPC14Controller.Qfun[0,2:]*dt,label='N4-initialized')
plt.plot(range(2,ShuffledLMPC21Controller.it), ShuffledLMPC21Controller.Qfun[0,2:]*dt,label='LMPC-N1-initialized')
plt.plot(range(2,ShuffledLMPC22Controller.it), ShuffledLMPC22Controller.Qfun[0,2:]*dt,label='LMPC-N2-initialized')
plt.plot(range(2,ShuffledLMPC23Controller.it), ShuffledLMPC23Controller.Qfun[0,2:]*dt,label='LMPC-N3-initialized')
#plt.plot(range(2,ShuffledLMPC24Controller.it), ShuffledLMPC24Controller.Qfun[0,2:]*dt,label='LMPC-N4-initialized')
plt.legend()


# ========================================= 1. Create and run LMPC controller 1 ========================================
raw_input("ENd -- running old thing now")

# This controller will start with a safe set consisting of PID and TVMPC laps
ClosedLoopShuffledLMPC1 = ClosedLoopData(dt, TimeLMPC, v0)
ShuffledLMPC1OpenLoopData = LMPCprediction(ShuffledN, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
ShuffledLMPC1Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
ShuffledLMPC1Controller = ControllerLMPC(numSS_Points, numSS_it, ShuffledN, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)

# add previously completed trajectories to Safe Set: 
ShuffledLMPC1Controller.addTrajectory(ClosedLoopDataShuffledPID)
ShuffledLMPC1Controller.addTrajectory(ClosedLoopDataShuffledPID)
#ShuffledLMPC1Controller.addTrajectory(ClosedLoopDataShuffledLTV_MPC)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopShuffledLMPC1.x[0,:]
x0_glob[0,:] = ClosedLoopShuffledLMPC1.x_glob[0,:]

for it in range(2, ShuffledLaps):
    #ShuffledLMPC1Controller.numSS_it = ShuffledLMPC1Controller.it
    ClosedLoopShuffledLMPC1.updateInitialConditions(x0, x0_glob)
    ShuffledLMPC1Simulator.Sim(ClosedLoopShuffledLMPC1, ShuffledLMPC1Controller, ShuffledLMPC1OpenLoopData) #this runs one lap at a time due to initialization!
    ShuffledLMPC1Controller.addTrajectory(ClosedLoopShuffledLMPC1)

    if ShuffledLMPC1Controller.feasible == 0:
        break
    else:
        #x0[0,:]      = ClosedLoopShuffledLMPC1.x[0,:]
        #x0_glob[0,:] = ClosedLoopShuffledLMPC1.x_glob[0,:]
        x0[0,:]      = ClosedLoopShuffledLMPC1.x[ClosedLoopShuffledLMPC1.SimTime, :] - np.array([0, 0, 0, 0, shuffledMap.TrackLength, 0])
        x0_glob[0,:] = ClosedLoopShuffledLMPC1.x_glob[ClosedLoopShuffledLMPC1.SimTime, :]


file_data = open('data/ShuffledLMPC1Controller.obj', 'wb')
pickle.dump(ClosedLoopLMPC, file_data)
pickle.dump(LMPController, file_data)
pickle.dump(LMPCOpenLoopData, file_data)
file_data.close()

plotClosedLoopLMPC(ShuffledLMPC1Controller, shuffledMap)
plt.show()

# ========================================= 3 Compare fastest lap times ================================================ 

#for i in range(0, ShuffledLMPC1Controller.it):
#    print("For PID-initialized LMPC Controller:")
#    print("Lap time at iteration ", i, " is ", ShuffledLMPC1Controller.Qfun[0, i]*dt, "s")
    
#for i in range(0, ShuffledLMPC2Controller.it):
 #   print("For RSS-initialized LMPC Controller:")
  #  print("Lap time at iteration ", i, " is ", ShuffledLMPC2Controller.Qfun[0, i]*dt, "s")

plt.figure()
plt.plot(range(2,ShuffledLMPC1Controller.it), ShuffledLMPC1Controller.Qfun[0,2:]*dt,label='PID-initialized')
plt.plot(range(2,ShuffledLMPC2Controller.it), ShuffledLMPC2Controller.Qfun[0,2:]*dt,label='RSS-initialized')
plt.legend()

# <codecell> Experiment about effect of N on PID-initialized system 

ClosedLoopShuffledLMPC1 = ClosedLoopData(dt, TimeLMPC, v0)
ShuffledLMPC1OpenLoopData = LMPCprediction(ShuffledN, n, d, TimeLMPC, numSS_Points, ShuffledLaps) #to store open-loop prediction and safe sets
ShuffledLMPC1Simulator = Simulator(shuffledMap, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON
ShuffledLMPC1Controller = ControllerLMPC(numSS_Points, numSS_it, ShuffledN, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, shuffledMap, ShuffledLaps, TimeLMPC, LMPC_Solver)

# add previously completed trajectories to Safe Set: 
ShuffledLMPC1Controller.addTrajectory(ClosedLoopDataShuffledPID)
ShuffledLMPC1Controller.addTrajectory(ClosedLoopDataShuffledPID)
#ShuffledLMPC1Controller.addTrajectory(ClosedLoopDataShuffledLTV_MPC)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopShuffledLMPC1.x[0,:]
x0_glob[0,:] = ClosedLoopShuffledLMPC1.x_glob[0,:]

for it in range(2, ShuffledLaps):
    #ShuffledLMPC1Controller.numSS_it = ShuffledLMPC1Controller.it
    ClosedLoopShuffledLMPC1.updateInitialConditions(x0, x0_glob)
    ShuffledLMPC1Simulator.Sim(ClosedLoopShuffledLMPC1, ShuffledLMPC1Controller, ShuffledLMPC1OpenLoopData) #this runs one lap at a time due to initialization!
    ShuffledLMPC1Controller.addTrajectory(ClosedLoopShuffledLMPC1)

    if ShuffledLMPC1Controller.feasible == 0:
        break
    else:
        #x0[0,:]      = ClosedLoopShuffledLMPC1.x[0,:]
        #x0_glob[0,:] = ClosedLoopShuffledLMPC1.x_glob[0,:]
        x0[0,:]      = ClosedLoopShuffledLMPC1.x[ClosedLoopShuffledLMPC1.SimTime, :] - np.array([0, 0, 0, 0, shuffledMap.TrackLength, 0])
        x0_glob[0,:] = ClosedLoopShuffledLMPC1.x_glob[ClosedLoopShuffledLMPC1.SimTime, :]


file_data = open('data/ShuffledLMPC1Controller.obj', 'wb')
pickle.dump(ClosedLoopLMPC, file_data)
pickle.dump(LMPController, file_data)
pickle.dump(LMPCOpenLoopData, file_data)
file_data.close()

plotClosedLoopLMPC(ShuffledLMPC1Controller, shuffledMap)
plt.show()
# ======================================================================================================================
# ========================================= PLOT TRACK/PREDICTIONS =====================================================
# ======================================================================================================================
# =============================================================================
# for i in range(0, LMPController.it):
#     print("Lap time at iteration ", i, " is ", LMPController.Qfun[0, i]*dt, "s")
# 
# raw_input("Finished LMPC - Start other plots?")
# print("===== Start Plotting")
# if animation_xyFlag == 1:
#     animation_xy(map, LMPCOpenLoopData, LMPController, 5)
#     # saveGif_xyResults(map, LMPCOpenLoopData, LMPController, 6)
# 
# if animation_stateFlag == 1:
#     animation_states(map, LMPCOpenLoopData, LMPController, 5)
#     # Save_statesAnimation(map, LMPCOpenLoopData, LMPController, 5)
# 
# if testCoordChangeFlag == 1:
#     unityTestChangeOfCoordinates(map, ClosedLoopDataPID)
#     unityTestChangeOfCoordinates(map, ClosedLoopDataLTI_MPC)
#     unityTestChangeOfCoordinates(map, ClosedLoopLMPC)
# 
# if plotOneStepPredictionErrors == 1:
#     it=5
#     onestep_errors = []
#     onestep_norm_errors = []
#     for i in range(1, int(LMPController.TimeSS[it])):
#         current_state = LMPController.SS[i, :, it]
#         current_pos = LMPController.SS[i, 4:6, it]
#         predicted_trajectory = LMPCOpenLoopData.PredictedStates[:, :, i-1, it]
#         predicted_state = predicted_trajectory[1,:]
#         onestep_errors.append(predicted_state-current_state)
#         onestep_norm_errors.append(np.linalg.norm(predicted_state-current_state))
# 
#     plt.figure();
#     onestep_errors = np.array(onestep_errors)
#     state_names = ['vx', 'vy', 'wz', 'epsi', 's', 'ey']
#     for state in range(onestep_errors.shape[1]):
#         if state == 0:
#             plt.title('One Step Prediction Error')
#         plt.subplot(onestep_errors.shape[1], 1, state+1)
#         plt.plot(onestep_errors[:,state])
#         plt.ylabel(state_names[state])
# 
# plt.show()
# 
# =============================================================================


