import random
import numpy as np

SIR_I = 2
SIR_C = 1
SIR_S = 0
SIR_R = -1

class Epidemic():
	def __init__(self, gridLength=2, epsilon=0, beta=1, CToI=1,
				timeRemaining=10, rewardForAnyNonI=False, initialInfectedAnywhere=True):
		# Epidemic parameters.
		self.epsilon = epsilon
		self.beta = beta
		self.CToI = CToI
		self.initialTimeRemaining = timeRemaining
		self.initialInfectedAnywhere = initialInfectedAnywhere
		self.rewardForAnyNonI = rewardForAnyNonI # For testing purposes.
		if gridLength < 0:
			raise ValueError("gridLength must be positive integer.")
		self.gridLength = gridLength # Number of hosts is gridLength squared.
		self.nHosts = gridLength**2
		self.nInitialInfected = 1
		self.nInitialSusceptible = self.nHosts - self.nInitialInfected
		self.reset()
	def reset(self):
		self.timeRemaining = self.initialTimeRemaining
		# Initialise host grid - just a list.
		self.hostGrid = [SIR_S] * self.nHosts
		# Choose initial host to be infected.
		if self.initialInfectedAnywhere:
			self.hostGrid[np.random.randint(0, self.nHosts)] = SIR_I
		else:
			self.hostGrid[0] = SIR_I
		return self.observe()
	def observe(self):
		# Cryptic hosts appear as Susceptible when observed.
		return [SIR_S if x < SIR_I else SIR_I for x in self.hostGrid]
	def step(self, action):
		# Apply the action and advance the epidemic one time step.
		# Update time remaining.
		self.timeRemaining -= 1
		# Copy hostGrid, in preparation for modifying it.
		newHostGrid = self.hostGrid[:]
		# Apply effect of action.
		# Whatever the hostGrid of the selected host was, set it to R#
		try:
			newHostGrid[action] = SIR_R
		except IndexError:
			print("Got action ",action,", but there are only ",self.nHosts," hosts.")
			# Action doesn't correspond to a host - do nothing.
			raise IndexError
		# Update hostState according to epidemic process.
		for host in range(self.nHosts):
			# S hosts can become C
			if newHostGrid[host] == SIR_S:
				# Primary infection
				if random.random() < self.epsilon:
					newHostGrid[host] = SIR_C
				# Secondary infection
				if self.getNumInfectedNeighbours(host) > 0:
					if random.random() < self.beta:
						newHostGrid[host] = SIR_C
			# C hosts can become I
			if newHostGrid[host] == SIR_C:
				if random.random() < self.CToI:
					newHostGrid[host] = SIR_I
		# Update host grid.
		self.hostGrid = newHostGrid
		# Create return values.
		observation = self.observe()
		# Epidemic is finished if time has run out.
		done = self.timeRemaining <= 0
		if done:
			# Reward based on number of S hosts.
			if self.rewardForAnyNonI:
				reward = sum(np.array(newHostGrid) != SIR_I)/self.nHosts # Higher reward for more S hosts
			else:
				reward = sum(np.array(newHostGrid) == SIR_S)/self.nHosts # Higher reward for more S hosts
		else:
			reward = 0
		info = None
		return observation, reward, done, info
	def getNumInfectedNeighbours(self, host):
		infectedNeighbours = 0
		for neighbourOffset in [-1, +1, -self.gridLength, +self.gridLength]:
			try:
				if self.hostGrid[host + neighbourOffset] >= SIR_C:
					infectedNeighbours += 1
			except:
				pass
		return infectedNeighbours
