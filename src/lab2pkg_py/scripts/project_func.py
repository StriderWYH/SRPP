#!/usr/bin/env python
import numpy as np
from scipy.linalg import expm, logm
from project_header import *
from cmath import asin, sin
from math import atan2
"""
Use 'expm' for matrix exponential.
Angles are in radian, distance are in meters.
"""




def Get_MS():
	# =================== Your code starts here ====================#
	# Fill in the correct values for w1~6 and v1~6, as well as the M matrix
	#M = np.eye(4)
	#S = np.zeros((6,6))
	w0 = np.array([0,0,1])
	w1 = np.array([0,1,0])
	w2 = np.array([0,1,0])
	w3 = np.array([0,1,0])
	w4 = np.array([1,0,0])
	w5 = np.array([0,1,0])

	q0 = np.array([-150,150,10])
	q1 = np.array([-150,270,162])
	q2 = np.array([94,270,162])
	q3 = np.array([307,177,162])
	q4 = np.array([307,260,162])
	q5 = np.array([390,260,162])
 
	s0 = np.array([w0,q0])
	s1 = np.array([w1,q1])
	s2 = np.array([w2,q2])
	s3 = np.array([w3,q3])
	s4 = np.array([w4,q4])
	s5 = np.array([w5,q5])

	v0 = -1 * np.cross(w0,q0)
	v1 = -1 * np.cross(w1,q1)
	v2 = -1 * np.cross(w2,q2)
	v3 = -1 * np.cross(w3,q3)
	v4 = -1 * np.cross(w4,q4)
	v5 = -1 * np.cross(w5,q5)
	v = np.array([v0,v1,v2,v3,v4,v5])
 
	S = np.array([[w0,v0],[w1,v1],[w2,v2],[w3,v3],[w4,v4],[w5,v5]])
	# w_b = np.array([[0,-w3,w2],[w3,0,-w1],[-w2,w1,0]])
	# S_b = np.array([[w_b,v],[0,0]])
	#s_b = np.zeros(())
	#s_b0 = np.array([[],[],[],[]])
	M = np.array([[1,0,0,390],[0,1,0,401],[0,0,1,215.5],[0,0,0,1]])




	# ==============================================================#
	return M, S

def calculate(x):
    w = np.array(x[0])
    v = np.array(x[1])
    s_b = np.zeros((4,4))
    s_b[0][0] = 0
    s_b[0][1] = -w[2]
    s_b[0][2] = w[1]
    s_b[0][3] = v[0]
    s_b[1][0] = w[2]
    s_b[1][1] = 0
    s_b[1][2] = -w[0]
    s_b[1][3] = v[1]
    s_b[2][0] = -w[1]
    s_b[2][1] = w[0]
    s_b[2][2] = 0
    s_b[2][3] = v[2]
    s_b[3][0] = 0
    s_b[3][1] = 0
    s_b[3][2] = 0
    s_b[3][3] = 0
    return s_b






"""
Function that calculates encoder numbers for each motor
"""
def lab_fk(theta1, theta2, theta3, theta4, theta5, theta6):

	# theta1 = theta1 - PI
	# theta4 = theta4 + PI/2
	# Initialize the return_value
	return_value = [None, None, None, None, None, None]
	# M,S = Get_MS()
	# print("Foward kinematics calculated:\n")

	# # =================== Your code starts here ====================#
	# T0 = expm(theta1*calculate(S[0]))
	# T1 = expm(theta2*calculate(S[1]))
	# T2 = expm(theta3*calculate(S[2]))
	# T3 = expm(theta4*calculate(S[3]))
	# T4 = expm(theta5*calculate(S[4]))
	# T5 = expm(theta6*calculate(S[5]))
	# T = np.matmul(T0,T1)
	# T = np.matmul(T,T2)
	# T = np.matmul(T,T3)
	# T = np.matmul(T,T4)
	# T = np.matmul(T,T5)
	# T = np.matmul(T,M)
	# print(str(T) + "\n")

	# ==============================================================#

	return_value[0] = theta1 + PI
	return_value[1] = theta2
	return_value[2] = theta3
	return_value[3] = theta4 - (0.5*PI)
	return_value[4] = theta5
	return_value[5] = theta6

	return return_value


"""
Function that calculates an elbow up Inverse Kinematic solution for the UR3
"""
def lab_invk(xWgrip, yWgrip, zWgrip, yaw_WgripDegree):
	# =================== Your code starts here ====================#
	xWgrip = xWgrip*1000
	yWgrip = yWgrip*1000
	zWgrip = zWgrip*1000
	x0grip = xWgrip + 150
	y0grip = yWgrip - 150
	z0grip = zWgrip - 10
	L1 = 152
	L2 = 120
	L3 = 244
	L4 = 93
	L5 = 213
	L6 = 83
	L7 = 83
	L8 = 82
	L9 = 53.5
	L10 = 59

	yaw_WgripDegree = yaw_WgripDegree * PI/180
	# center
	x_cen = x0grip - L9 * np.cos(yaw_WgripDegree)
	y_cen = y0grip - L9 * np.sin(yaw_WgripDegree)
	z_cen = z0grip
	# theta 1 
	theta1 = np.arcsin(-(L2-L4+L6)/(np.sqrt(x_cen*x_cen+y_cen*y_cen)))+atan2(y_cen,x_cen)
	# theta1 = theta1 * 180/PI
	# theta 6
	theta6 = theta1 + PI/2 - yaw_WgripDegree
	# 3end
	x3end = x_cen + (L4 + 27) * np.sin(theta1) - L7 * np.cos(theta1)
	y3end = y_cen - (L4 + 27) * np.cos(theta1) - L7 * np.sin(theta1)
	z3end = z_cen + L10 + L8
	# theta 2
	d1 =np.sqrt((x3end*x3end)+(y3end*y3end)+(z3end-L1)*(z3end-L1))
	alpha = np.arcsin((z3end-L1)/d1)
	theta2 = -(np.arccos(((L3*L3)+(d1*d1)-(L5*L5))/(2*L3*d1))+alpha)
	# theta2 = theta2 *180/PI
	#theta 4
	theta4 = -(np.arccos((L5*L5+d1*d1-L3*L3)/(2*L5*d1))-alpha)
	# theta4 = theta4 *180/PI
	#theta 3
	theta3 = -theta2 - theta4
	#theta5
	theta5 = -PI/2
 
	# print("theta2\n")
	# print(theta2)
	# print("theta3\n")
	# print(theta3)
	#theta1 = theta1 * PI/180
	#theta2 = theta2 * PI/180
	#theta3 = theta3 * PI/180
	#theta4 = theta4 * PI/180
	#theta5 = theta5 * PI/180
	#theta6 = theta6 * PI/180

	

	# ==============================================================#
	return lab_fk(theta1, theta2, theta3, theta4, theta5, theta6)
