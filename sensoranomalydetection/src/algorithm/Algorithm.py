import numpy as np


class AnomalyDetectionAlgorithmInterface:

	def fit(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
		raise NotImplementedError

	def predict(self, inputs: list) -> (bool, bool):
		raise NotImplementedError
