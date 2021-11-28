import pickle

import numpy as np
import neat

from src.algorithm.Algorithm import AnomalyDetectionAlgorithmInterface
from src.algorithm.visualize import draw_net


class NEATAlgorithm(AnomalyDetectionAlgorithmInterface):

	def __init__(self, config_route: str):
		"""Initialize variables"""
		self.config_route = config_route
		self.config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
		                                 neat.DefaultStagnation, self.config_route)
		self.training_inputs = None
		self.training_outputs = None
		self.winner = None
		self.winner_net = None

	def visualize_net(self, g):
		"""Draw the neural net on the file genome.svg"""
		draw_net(self.config, genome=g, filename="genome", node_names={
			0: "is_drift",
			1: "is_dangerous_drift",
			-1: "water_value",
			-2: "air_value",
			-3: "amoni_value"
		})

	def save_genome_into_file(self, g):
		"""Save a genome into a file called winner.pkl"""
		with open("winner.pkl", "wb") as f:
			pickle.dump(g, f)

	def load_genome(self):
		"""Load a genome from a file called winner.pkl"""
		with open("winner.pkl", "rb") as f:
			self.winner = pickle.load(f)

	def generation(self, genomes, config):
		"""The function executed in every generation"""
		nets = []
		ge = []

		for _, g in genomes:
			# Create a neural network
			net = neat.nn.FeedForwardNetwork.create(g, config)
			nets.append(net)
			# Create a genome with fitness 0.0
			g.fitness = 0
			ge.append(g)

		fitness_max = 0
		rows = 0
		# For all available training inputs, use a network to guess the output
		for row_index, row in enumerate(self.training_inputs):
			for ind, net in enumerate(nets):
				net_outputs = net.activate(row)
				parsed_net_outputs = [1 if elem >= 0.5 else 0 for elem in net_outputs]
				for out_index, elem in enumerate(parsed_net_outputs):
					if elem == self.training_outputs[row_index][out_index]:
						# If we got the input right, have more fitness
						ge[ind].fitness += 1
			rows += 1
			fitness_max += 2

		# Transform the fitness into a percentage
		for g in ge:
			g.fitness = (g.fitness * 100) / fitness_max

		# Save the winner of the current generation into the file
		max_fitness = max([g.fitness for g in ge])
		for g in ge:
			if g.fitness == max_fitness:
				self.winner_net = neat.nn.FeedForwardNetwork.create(g, self.config)
				self.save_genome_into_file(g)
				self.visualize_net(g)

	def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
		"""Given inputs and their outputs, train a NN"""
		self.training_inputs = inputs
		self.training_outputs = outputs

		# Population
		p = neat.Population(self.config)

		p.add_reporter(neat.StdOutReporter(True))
		stats = neat.StatisticsReporter()
		p.add_reporter(stats)

		self.winner = p.run(self.generation, 20)
		self.winner_net = neat.nn.FeedForwardNetwork.create(self.winner, self.config)
		self.save_genome_into_file(self.winner)
		self.visualize_net(self.winner)

	def predict(self, inputs: list) -> (bool, bool):
		"""Once the NN is trained, with a given input predict the output"""
		if self.winner_net is None:
			try:
				self.load_genome()
				self.winner_net = neat.nn.FeedForwardNetwork.create(self.winner, self.config)
			except FileNotFoundError:
				print("ERROR: Algorithm not yet trained")
				return
		outs = self.winner_net.activate(inputs)
		return outs[0] >= 0.5, outs[1] >= 0.5
