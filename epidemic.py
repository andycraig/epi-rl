import random
import numpy as np
from OrderedEnum import OrderedEnum

class SIR(OrderedEnum):
	I = 2
	C = 1
	S = 0
	R = -1
	def __repr__(self):
		# Just return the compartment name.
		return super(SIR, self).__repr__()[5:6]

class Epidemic():
	def __init__(self, gridLength=2, epsilon=0, beta=1, CToI=1, timeRemaining=10, rewardForAnyNonI=False):
		# Epidemic parameters.
		self.epsilon = epsilon
		self.beta = beta
		self.CToI = CToI
		self.timeRemaining = timeRemaining
		self.rewardForAnyNonI = rewardForAnyNonI # For testing purposes.
		self.gridLength = gridLength # Number of hosts is gridLength squared.
		self.nHosts = gridLength**2
		self.nInitialInfected = 1
		self.nInitialSusceptible = self.nHosts - self.nInitialInfected
		self.reset()
	def reset(self):
		# Initialise host grid - just a list with an infected at the corner.
		self.hostGrid = [SIR.I] + [SIR.S] * self.nInitialSusceptible
		return self.observe()
	def observe(self):
		# Cryptic hosts appear as Susceptible when observed.
		return [SIR.S if x < SIR.I else SIR.I for x in self.hostGrid]
	def step(self, action):
		# Apply the action and advance the epidemic one time step.
		# Update time remaining.
		self.timeRemaining -= 1
		# Copy hostGrid, in preparation for modifying it.
		newHostGrid = hostGrid[:]
		# Apply effect of action.
		# Whatever the hostGrid of the selected host was, set it to R#
		try:
			newHostGrid[action] = SIR.R
		except KeyError:
			# Action doesn't correspond to a host - do nothing.
			pass
		# Update hostState according to epidemic process.
		for host in range(nHosts):
			# S hosts can become C
			if newHostGrid[host] == SIR.S:
				# Primary infection
				if random.random() < self.epsilon:
					newHostGrid[host] = SIR.C
				# Secondary infection
				if getNumInfectedNeighbours(newHostGrid, host) > 0:
					if random.random() < self.beta:
						newHostGrid[host] = SIR.C
			# C hosts can become I
			if newHostGrid[host] == SIR.C:
				if random.random() < self.CToI:
					newHostGrid[host] = SIR.I
		# Update host grid.
		self.hostGrid = newHostGrid
		# Create return values.
		observation = self.observe()
		# Epidemic is finished if time has run out.
		done = self.timeRemaining <= 0
		if done:
			# Reward based on number of S hosts.
			if self.rewardForAnyNonI:
				reward = sum(np.array(newHostGrid) != SIR.I)/nHosts # Higher reward for more S hosts
			else:
				reward = sum(np.array(newHostGrid) == SIR.S)/nHosts # Higher reward for more S hosts
		else:
			reward = 0
		info = None
		return observation, reward, done, info
	def getNumInfectedNeighbours(self, host):
		infectedNeighbours = 0
		for neighbourOffset in [-1, +1, -self.gridLength, +self.gridLength]:
			try:
				if self.hostGrid[host + neighbourOffset] >= SIR.C:
					infectedNeighbours += 1
			except:
				pass
		return infectedNeighbours
