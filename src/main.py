import pandas as pd

from algorithm.NEATAlgorithm import NEATAlgorithm
from algorithm.RecurrentNEAT import RecurrentNEAT


def get_input_data(water_route: str, air_file: str, amoni_file: str):
	# Load data
	data_water = pd.read_csv(water_route)
	data_air = pd.read_csv(air_file)
	data_amoni = pd.read_csv(amoni_file)
	# Join Data
	result = data_water.merge(data_air, on='row_date', how='outer', suffixes=('_air', '_water'))
	result = result.merge(data_amoni, on='row_date').dropna()
	# Split data
	inputs = result[['value_water', 'value_air', 'value']]
	outputs = result[['is_drift', 'dangerous_drift']].astype(int)
	return inputs.to_numpy(), outputs.to_numpy()


if __name__ in "__main__":
	model = NEATAlgorithm("algorithm/neat-config.txt")
	inputs, outputs = get_input_data("../data/aigua.csv", "../data/aire.csv", "../data/amoni.csv")
	model.fit(inputs, outputs)
