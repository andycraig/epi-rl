import random
import numpy as np

SIR_I = 2
SIR_C = 1
SIR_S = 0
SIR_R = -1

class Epidemic():
	def __init__(self, gridLength=2, epsilon=0, beta=1, CToI=1,
				timeRemaining=10, rewardForC=False, rewardForR=False,
				initialInfectedAnywhere=True,
				initiallyCryptic=False):
		# Epidemic parameters.
		self.epsilon = epsilon
		self.beta = beta
		self.CToI = CToI
		self.initialTimeRemaining = timeRemaining
		self.initialInfectedAnywhere = initialInfectedAnywhere
		self.rewardForC = rewardForC # For testing purposes.
		self.rewardForR = rewardForR
		if gridLength < 0:
			raise ValueError("gridLength must be positive integer.")
		self.gridLength = gridLength # Number of hosts is gridLength squared.
		self.nHosts = gridLength**2
		self.nInitialInfected = 1
		self.nInitialSusceptible = self.nHosts - self.nInitialInfected
		self.initiallyCryptic = initiallyCryptic
		self.reset()
	def reset(self):
		self.timeRemaining = self.initialTimeRemaining
		# Initialise host grid - just a list.
		self.hostGrid = [SIR_S] * self.nHosts
		if self.initiallyCryptic:
			initialInfectionStatus = SIR_C
		else:
			initialInfectionStatus = SIR_I
		# Choose initial host to be infected.
		if self.initialInfectedAnywhere:
			self.hostGrid[np.random.randint(0, self.nHosts)] = initialInfectionStatus
		else:
			self.hostGrid[0] = initialInfectionStatus
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
			# Action doesn't correspond to a host - do nothing.
			pass
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
		done = self.isDone()
		reward = self.getReward()
		info = None
		return observation, reward, done, info
	def isDone(self):
		# Epidemic is finished if time has run out.
		return self.timeRemaining <= 0
	def getReward(self):
		if self.isDone():
			unnormalisedReward = sum(np.array(self.hostGrid) == SIR_S)
			# Reward based on number of S hosts.
			if self.rewardForC:
				unnormalisedReward += sum(np.array(self.hostGrid) == SIR_C) # Higher reward for more S hosts
			if self.rewardForR:
				unnormalisedReward += sum(np.array(self.hostGrid) == SIR_R) # Higher reward for more S hosts
			reward = 1.0 * unnormalisedReward / self.nHosts
		else:
			reward = 0
		return reward
	def getNumInfectedNeighbours(self, host):
		infectedNeighbours = 0
		for neighbourOffset in [-1, +1, -self.gridLength, +self.gridLength]:
			try:
				if self.hostGrid[host + neighbourOffset] >= SIR_C:
					infectedNeighbours += 1
			except:
				pass
		return infectedNeighbours
	def getRandomAction(self):
		return np.random.choice(self.nHosts + 1)
	@staticmethod
	def outputSample(fileName="sampleEpidemic.txt"):
		timeRemaining = 10
		s = Epidemic(gridLength=5, epsilon=0, beta=0.25, CToI=1, timeRemaining=timeRemaining)
		with open(fileName, 'w') as f:
			for t in range(timeRemaining):
				f.write(str(s))
				s.step(s.nHosts - 1) # Always rogue last host.
		print("Wrote sample epidemic to ", fileName)
	def __repr__(self):
		return "Time remaining: " + str(self.timeRemaining) + "\n" + str(np.reshape(self.hostGrid, [self.gridLength, self.gridLength])) + "\n"
