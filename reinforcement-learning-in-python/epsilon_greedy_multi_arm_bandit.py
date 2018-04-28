import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

"""
Scenario: Multi-Arm-Bandit
Each bandit has some inherent probability for a positive outcome: '1', vs negative outcome: '0'
Our agent does not know these probabilities

We are interested in finding the explore/exploit strategy that maximizes profit
"""

class Bandit():
	def __init__(self, true_p):
		self.true_p = true_p
		self.predicted_p = 0
		self.num_pulls_made = 0

	def pull(self):
		if np.random.rand() <= self.true_p:
			return 1
		else:
			return 0

	def update_predicted_p(self, current_pull_result):
		self.num_pulls_made += 1
		self.predicted_p = (1 - 1.0/self.num_pulls_made)*self.predicted_p + \
							  (1.0/self.num_pulls_made)*current_pull_result


K_RUNS = 30 # number of runs to average each result over
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
EPSILONS = [0.001, 0.01, 0.05, 0.1, 0.3]
BANDIT_MEANS = [0, 0.3, 0.5, 0.7]
NUM_TURNS = 1000

def explore(bandits):
	bandit = np.random.choice(bandits, size=1)[0]
	return play_single_turn(bandit)

def exploit(bandits):
	bandit = bandits[np.argmax([b.predicted_p for b in bandits])]
	return play_single_turn(bandit)

def play_single_turn(bandit):
	result = bandit.pull()
	bandit.update_predicted_p(result)
	return result

def epsilon_greedy_turn(epsilon, bandits):
	if np.random.rand() < epsilon:
		return explore(bandits)
	else:
		return exploit(bandits)

def play_entire_game(bandits, num_turns, epsilon):
	sum_profits = 0.0
	for i in range(1, num_turns):
		sum_profits += epsilon_greedy_turn(epsilon, bandits)
	return sum_profits

def compare_epsilons(epsilons, bandit_true_ps, max_turns):
	for epsilon_index, epsilon in enumerate(epsilons):
		average_profit_per_num_turns = []
		turns = range(1, max_turns)
		for num_turns in turns:
			sum_profits = 0.0
			for i in range(K_RUNS):
				bandits = [Bandit(p) for p in bandit_true_ps]
				sum_profits += play_entire_game(bandits, num_turns, epsilon)

			average_profit_per_num_turns.append(sum_profits/(num_turns*K_RUNS))
		plt.plot(turns, average_profit_per_num_turns, color=COLORS[epsilon_index])
	plt.legend(["epsilon = " + str(epsilon) for epsilon in epsilons])	
	plt.show()

compare_epsilons(EPSILONS, BANDIT_MEANS, NUM_TURNS)
