import numpy as np
import neat
from src.algorithm import visualize

from src.algorithm.Algorithm import AnomalyDetectionAlgorithmInterface


class NEATAlgorithm(AnomalyDetectionAlgorithmInterface):
	training_inputs = None
	training_outputs = None
	genome_config = None
	winner = None

	def __init__(self, config_route: str):
		self.config_route = config_route

	def visualize(self, winner):
		visualize.draw_net(self.genome_config, genome=winner, filename="winner_network")

	def generation(self, genomes, config):
		self.genome_config = config

		nets = []
		ge = []

		for _, g in genomes:
			# Create a neural network
			net = neat.nn.FeedForwardNetwork.create(g, config)
			nets.append(net)
			# Create a genome with fitness 0.0
			g.fitness = 0
			ge.append(g)

		for row_index, row in enumerate(self.training_inputs):
			for ind, net in enumerate(nets):
				net_outputs = net.activate(row)
				parsed_net_outputs = [1 if elem >= 0 else 0 for elem in net_outputs]
				for out_index, elem in enumerate(parsed_net_outputs):
					if elem == self.training_outputs[row_index][out_index]:
						ge[ind].fitness += 1

		for ind, net in enumerate(nets):
			ge[ind].fitness = ge[ind].fitness / len(self.training_outputs)

	def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
		self.training_inputs = inputs
		self.training_outputs = outputs

		config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
		                            neat.DefaultStagnation, self.config_route)

		# Population
		p = neat.Population(config)

		p.add_reporter(neat.StdOutReporter(True))
		stats = neat.StatisticsReporter()
		p.add_reporter(stats)

		self.winner = p.run(self.generation, 10)
		self.visualize(self.winner)

	def predict(self, inputs: list) -> (bool, bool):
		raise NotImplementedError
