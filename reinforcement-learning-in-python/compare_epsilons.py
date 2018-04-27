# import pdb; pdb.set_trace()
import numpy as np
import matplotlib as plt

"""
Trying out different epsilons in an 
Epsilon-Greedy Multi-Arm-Bandit scenario
"""

class Bandit():
	def __init__(self, true_mean):
		self.true_mean = true_mean
		self.predicted_mean = 0
		self.num_iterations_passed = 0

	def pull(self):
		return np.random.rand() + self.true_mean

	def update_predicted_mean(self, current_predicted_mean):
		self.num_iterations_passed += 1
		self.predicted_mean = (1 - 1.0/self.num_iterations_passed)*self.predicted_mean + (1.0/self.num_iterations_passed)*current_predicted_mean


def compare_epsilons(epsilons, bandit_means):



epsilons = [0.1]
bandit_means = [0]
compare_epsilons(epsilons, bandit_means)
