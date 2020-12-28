import numpy as np
import matplotlib.pyplot as plt
import math
import random
random.seed(20)
class EKF():
	def __init__(self, x0, P0, Q, R, dt):
		self.dt = dt
		### initial state vector: (vehicle state(4) + landmakr state(2))
		self.x = x0.copy()

		self.P = P0
		### process noise: (vehicle state(4)+ landmark state(2) * N) x (vehicle state(4)+ landmark state(2) * N)  
		self.Q = Q 
		### measurement noise: (landmark state(2) * N) x (vehicle state(4) + landmark state(2) * N)
		self.R = R 
		

		# initiate estimate 
		kkk = math.sqrt(self.P[4,4]) * np.random.randn()
		self.x[4:] += kkk # landmarks
		print(kkk)
		self.landmark_count = int((self.x.shape[0] - 4) / 2)

	def predict(self):
		### State transition matrix
		self.F = np.eye(self.x.shape[0])
		self.F[0,3] = math.cos(self.x[2])*self.dt # x += V*cos(yaw)*dt
		self.F[1,3] = math.sin(self.x[2])*self.dt # y += V*sin(yaw)*dt

		### Predict state (xhat)
		self.x = np.matmul(self.F, self.x)
		# print("Predict: x")
		# print(self.x)

		### Predict observation (zhat)
		self.z = []
		self.z.append(self.x[2]) # add yaw to the measurement
		self.z.append(self.x[3]) # add velocity to the measurement
		for p in self.x[4:].reshape(self.landmark_count, 2):
			ranging = math.sqrt((p[0]-self.x[0])**2 + (p[1]-self.x[1])**2) 
			bearing = math.atan2(p[1]-self.x[1],p[0]-self.x[0]) - self.x[2]
			self.z.append(ranging)
			self.z.append(bearing)
		self.z = np.array(self.z)

		# Propagate state covariance
		self.P = np.matmul(np.matmul(self.F, self.P),np.transpose(self.F)) + self.Q * self.dt
		# print("Propagate P")
		# print(self.P)

	def update(self, z):
		# Jacobian of H
		H = self.linearize_H()
		# Find gain matrix
		v = z - self.z
		# print("observation: " + str(z))
		# print("prediction: " + str(self.z))
		S = np.matmul(np.matmul(H, self.P), np.transpose(H)) + self.R / self.dt
		W = np.matmul(np.matmul(self.P, np.transpose(H)), np.linalg.inv(S))

		# Update based on the gain matrix
		self.x = self.x + np.matmul(W, v)
		# np.set_printoptions(precision=2)
		# print("S: ")
		# print(S)
		# print("W: ")
		# print(W)
		# print("v: ")
		# print(v)
		# print(np.matmul(W,v))
		self.P = self.P - np.matmul(W,np.matmul(S,np.transpose(W)))

	def linearize_H(self):
		# H: 12 x 14 matrix
		# LiDAR measurement model
		H = np.zeros((self.landmark_count*2, self.x.shape[0]))
		for i in range(self.landmark_count):
			H[2*i, 0] = (self.x[0] - self.x[4+2*i]) / np.linalg.norm(np.array([self.x[0]-self.x[4+2*i], self.x[1]-self.x[5+2*i]]))
			H[2*i, 1] = (self.x[1] - self.x[5+2*i]) / np.linalg.norm(np.array([self.x[0]-self.x[4+2*i], self.x[1]-self.x[5+2*i]]))
			H[2*i, 4+2*i] = -H[2*i, 0]
			H[2*i, 5+2*i] = -H[2*i, 1]
			H[2*i+1,0] = (self.x[5+2*i] - self.x[1]) / np.linalg.norm(np.array([self.x[0]-self.x[4+2*i], self.x[1]-self.x[5+2*i]]))**2
			H[2*i+1,1] = (self.x[0] - self.x[4+2*i]) / 		self.P[0,0] = 0.0
		self.P[1,1] = 0.0	

class Simulation()
	def __init__(self, dt, sim_time, max_range, ranging_std, bearing_std):
		self.dt = dt
		self.sim_time = sim_time
		self.max_range = max_range
		self.ranging_std = ranging_std
		self.bearing_std = bearing_std

	def init_params(self, x, y, yaw, vel, landmarks):
		self.landmarks = landmarks
		vec_size = 2 * len(self.landmarks) + 4
		self.Q = np.eye(vec_size)
		self.R = np.eye(2*(len(landmarks)+1))
		self.x = np.concatenate((np.array([x, y, yaw, vel]), np.array(self.landmarks).flatten()))
		self.P = np.eye(vec_size)*10.0
		self.P[0,0] = 0.0
		self.P[1,1] = 0.0
		for i in range(len(self.R[0])):
			if i >= 2:
				if i % 2 == 0:
					self.R[i,i] = self.ranging_std ** 2
				else:
					self.R[i,i] = self.bearing_std ** 2
			else:
				self.R[i,i] = 0.0
		# print(self.R)

	def move(self):
		self.x[0] += self.x[3]*math.cos(self.x[2])*self.dt
		self.x[1] += self.x[3]*math.sin(self.x[2])*self.dt	

	def observe(self):
		z = []
		z.append(self.x[2]) # Assume no noise in measuring yaw
		z.append(self.x[3]) # Assume no noise in measuring velocity
		for p in self.landmarks:
			ranging = math.sqrt((p[0]-self.x[0])**2 + (p[1]-self.x[1])**2) + math.sqrt(self.R[2,2]) * np.random.randn()
			bearing = math.atan2(p[1]-self.x[1],p[0]-self.x[0]) - self.x[2] + math.sqrt(self.R[3,3]) * np.random.randn()
			z.append(ranging)
			z.append(bearing)
		return np.array(z)

	def start(self):
		time = 0.0
		ekf = EKF(self.x, self.P, self.Q, self.R, self.dt)
		# visualize GT - landmarks
		for i in range(len(self.landmarks)):
			plt.plot(self.x[4+2*i],self.x[5+2*i],'x', color='green', zorder=100)
			plt.plot(ekf.x[0], ekf.x[1], 'o', color='red')
			plt.plot(self.x[0],self.x[1], 'o', color='black')
		while time <= self.sim_time:
			# print("Time: " + str(time) + " s")
			self.move() # GT pose change
			# print("GT")
			# print(self.x)
			obs = self.observe()
			ekf.predict()
			ekf.update(obs)
			# print("Update: x")
			# print(ekf.x)
			plt.plot(ekf.x[0], ekf.x[1], '.', color='red')
			plt.plot(self.x[0],self.x[1], '.', color='black')
			for i in range(len(self.landmarks)):
				plt.plot(ekf.x[4+2*i],ekf.x[5+2*i], 'x', color='yellow')
			time += self.dt
			# plt.show()

		plt.axis([-5,10,-5,10])
		plt.show()


def main():
	landmarks = [[5, 8],[7, 4], [1, 5], [5, 3], [8, 1]]
	# Start simulation
	sim = Simulation(dt=0.1, sim_time=20, max_range=10.0, ranging_std=0.1, bearing_std=0.1)
	sim.init_params(x=0, y=0, yaw=90*math.pi/180.0, vel=0.5, landmarks=landmarks)
	sim.start()
main()