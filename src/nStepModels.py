#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 11:39:13 2018

@author: vallon2
"""

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
from LMPC import ControllerLMPC, PWAControllerLMPC,ControllerLTI_LMPC,ControllerLTI_LMPC_NStep
from Utilities import Regression, nStepRegression
from plot import plotTrajectory, plotClosedLoopLMPC, animation_xy, animation_states, saveGif_xyResults, Save_statesAnimation, plotMap, plotSafeSet
import numpy as np
import matplotlib.pyplot as plt
import pdb
import pickle

# ======================================================================================================================
# ============================ Choose which controller to run to set up problem ========================================
# ======================================================================================================================
RunPID     = 1; plotFlag       = 1
RunMPC     = 1; plotFlagMPC    = 1
RunMPC_tv  = 0; plotFlagMPC_tv = 0
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
N          = 12           # Horizon length (12)
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

# <codecell> 

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
    
if plotFlag == 1:
    plotTrajectory(map, ClosedLoopDataPID.x, ClosedLoopDataPID.x_glob, ClosedLoopDataPID.u)
    plt.show()
    
# ======================================================================================================================
# ======================================  LINEAR REGRESSION ============================================================
# ======================================================================================================================
raw_input("Finished PID - Start LTI Tracking MPC?")
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

if plotFlagMPC == 1:
    plotTrajectory(map, ClosedLoopDataLTI_MPC.x, ClosedLoopDataLTI_MPC.x_glob, ClosedLoopDataLTI_MPC.u)
    plt.show()

# ======================================================================================================================
# ==============================  LMPC w\ LTI REGRESSION-===============================================================
# ======================================================================================================================
raw_input("Finished LTI Tracking MPC - Start LTI-LMPC?")
ClosedLoopLMPC = ClosedLoopData(dt, TimeLMPC, v0)
LMPCOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, Laps) #to store open-loop prediction and safe sets
LMPCSimulator = Simulator(map, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON

if runPWAFlag == 1:
    LMPController = PWAControllerLMPC(10, numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)
else:
    LMPController = ControllerLTI_LMPC(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)

# add previously completed trajectories to Safe Set: 
LMPController.addTrajectory(ClosedLoopDataPID)
LMPController.addTrajectory(ClosedLoopDataLTI_MPC)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = ClosedLoopLMPC.x[0,:]
x0_glob[0,:] = ClosedLoopLMPC.x_glob[0,:]

if RunLMPC == 1:
    for it in range(2, Laps):
        ClosedLoopLMPC.updateInitialConditions(x0, x0_glob)
        LMPCSimulator.Sim(ClosedLoopLMPC, LMPController, LMPCOpenLoopData) #this runs one lap at a time due to initialization!
        LMPController.addTrajectory(ClosedLoopLMPC)

        if LMPController.feasible == 0:
            break
        else:
            # Reset Initial Conditions
#            x0[0,:]      = ClosedLoopLMPC.x[ClosedLoopLMPC.SimTime, :] - np.array([0, 0, 0, 0, map.TrackLength, 0])
#            x0_glob[0,:] = ClosedLoopLMPC.x_glob[ClosedLoopLMPC.SimTime, :]
            x0[0,:]      = ClosedLoopLMPC.x[0,:]
            x0_glob[0,:] = ClosedLoopLMPC.x_glob[0,:]

    #file_data = open(sys.path[0]+'/data/LMPController.obj', 'wb')
    file_data = open('data/LMPController.obj', 'wb')
    pickle.dump(ClosedLoopLMPC, file_data)
    pickle.dump(LMPController, file_data)
    pickle.dump(LMPCOpenLoopData, file_data)
    file_data.close()
    
else:
    #file_data = open(sys.path[0]+'/data/LMPController.obj', 'rb')
    file_data = open('data/LMPController.obj', 'rb')
    ClosedLoopLMPC = pickle.load(file_data)
    LMPController  = pickle.load(file_data)
    LMPCOpenLoopData  = pickle.load(file_data)
    file_data.close()

if plotFlagLMPC == 1:
    plotClosedLoopLMPC(LMPController, map)
    plt.show()

# plot the safe set along the map 
raw_input("LMPC with LTI is done.")


plt.figure()
plt.plot(range(0,LMPController.it), LMPController.Qfun[0,0:6]*dt)


# <codecell> 

# ======================================================================================================================
# ==================================  N-STEP LINEAR REGRESSION =========================================================
# ======================================================================================================================
raw_input("Finished LTI LMPC - Start nStep Tracking MPC?")
lamb = 0.0000001
#fit linear dynamics to the closed loop data: x2 = A*x1 + b*u1; lamb is weight on frob norm of W
Theta, nStepError = nStepRegression(ClosedLoopDataPID.x, ClosedLoopDataPID.u, N, lamb)

# ======================================================================================================================
# ==================================  N-STEP TRACKING MPC ==============================================================
# ======================================================================================================================





# ======================================================================================================================
# ==============================  LMPC w\ N-STEP REGRESSION-============================================================
# ======================================================================================================================
raw_input("Finished TI-MPC - Start N-step LMPC?")
print("Starting N-step LMPC")

nClosedLoopLMPC = ClosedLoopData(dt, TimeLMPC, v0)
nLMPCOpenLoopData = LMPCprediction(N, n, d, TimeLMPC, numSS_Points, Laps) #to store open-loop prediction and safe sets
nLMPCSimulator = Simulator(map, 1, 1) #now this simulator only runs for one lap, with the LMPC flag ON

nStepLMPController = ControllerLTI_LMPC_NStep(numSS_Points, numSS_it, N, Qslack, Q_LMPC, R_LMPC, dR_LMPC, n, d, shift, dt, map, Laps, TimeLMPC, LMPC_Solver)

nStepLMPController._getTheta(ClosedLoopDataPID.x, ClosedLoopDataPID.u, N, lamb)

# add previously completed trajectories to Safe Set: 
nStepLMPController.addTrajectory(ClosedLoopDataPID)
nStepLMPController.addTrajectory(ClosedLoopDataPID)
#nStepLMPController.addTrajectory(ClosedLoopDataLTI_MPC)

x0           = np.zeros((1,n))
x0_glob      = np.zeros((1,n))
x0[0,:]      = nClosedLoopLMPC.x[0,:]
x0_glob[0,:] = nClosedLoopLMPC.x_glob[0,:]

if RunLMPC == 1:
    for it in range(2, Laps):
        nClosedLoopLMPC.updateInitialConditions(x0, x0_glob)
        nLMPCSimulator.Sim(nClosedLoopLMPC, nStepLMPController, nLMPCOpenLoopData) #this runs one lap at a time due to initialization!
        nStepLMPController.addTrajectory(nClosedLoopLMPC)

        if nStepLMPController.feasible == 0:
            break
        else:
            # Reset Initial Conditions
#            x0[0,:]      = ClosedLoopLMPC.x[ClosedLoopLMPC.SimTime, :] - np.array([0, 0, 0, 0, map.TrackLength, 0])
#            x0_glob[0,:] = ClosedLoopLMPC.x_glob[ClosedLoopLMPC.SimTime, :]
            x0[0,:]      = nClosedLoopLMPC.x[0,:]
            x0_glob[0,:] = nClosedLoopLMPC.x_glob[0,:]

    #file_data = open(sys.path[0]+'/data/LMPController.obj', 'wb')
    file_data = open('data/nLMPController.obj', 'wb')
    pickle.dump(nClosedLoopLMPC, file_data)
    pickle.dump(nStepLMPController, file_data)
    pickle.dump(nLMPCOpenLoopData, file_data)
    file_data.close()
    
else:
    #file_data = open(sys.path[0]+'/data/LMPController.obj', 'rb')
    file_data = open('data/LMPController.obj', 'rb')
    ClosedLoopLMPC = pickle.load(file_data)
    LMPController  = pickle.load(file_data)
    LMPCOpenLoopData  = pickle.load(file_data)
    file_data.close()

if plotFlagLMPC == 1:
    plotClosedLoopLMPC(nStepLMPController, map)
    plt.show()

raw_input("LMPC with N-step model is done.")

plt.figure()
plt.plot(range(2,nStepLMPController.it), nStepLMPController.Qfun[0,2:]*dt)
