import tensorflow as tf
import numpy as np

def discount_rewards(r, gamma):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(range(len(r))):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

def softmax(x):
	exps = np.exp(x)
	return exps / exps.sum(axis=0)

def getAction(logits):
	# y is
	# tfprob and y should match up.
	# tfprob [1.] should give action = 0, y = [1]
	# tfprob [0.] should give action = 1, y = [0]
	# TODO Change from np.random.multinomial, which is very slow.
	pvals = softmax(logits[0])
	try:
		y = np.random.multinomial(n=1, pvals=pvals)
	except ValueError as e:
		# Sometimes if one probability is very large, numerical imprecision
		# results in sum(pvals) > 1 and there is a ValueError.
		# In this case, just take the largest value.
		action, pmax = max(enumerate(pvals), key=lambda x: x[1])
		if pmax < 0.75:
			print(pvals)
			raise ValueError("There was a ValueError, and largest prob was " + str(pmax) + " < 0.75.")
		else:
			print("Deterministic action to avoid error.")
			# y is one-hot of action.
			y = np.eye(len(pvals))[action,]
	try:
		action = np.asscalar(np.where(y == 1)[0])
	except ValueError:
		print("ValueError! y was ", y)
	return action, y

def output(env, logits, observations_placeholder, fileName, sess):
	with open(fileName, "w") as f:
		for iOutput in range(2):
			observation = env.reset()
			done = False
			f.write(" ====== EXAMPLE RUN " + str(iOutput) + " ======\n")
			while not done:
				f.write("\n" + str(env))
				# Get observation and choose action.
				x = np.reshape(observation,[1,len(observation)])
				tflogits = sess.run(logits,feed_dict={observations_placeholder: x})
				action, y = getAction(tflogits)
				f.write("Took action: " + str(action) + "\n")
				observation, reward, done, info = env.step(action)
				f.write("Got reward: " + str(reward) + "\n")
			observation = env.reset()
			f.write("\n\n")
