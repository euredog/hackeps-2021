'''
Authorship: Joel Aumedes, Alejandro Clavera, Moisés Bernaus and Marc Cervera
'''

import pandas as pd

from algorithm.NEATAlgorithm import NEATAlgorithm
from config import neat_config, water_path, air_path, ammonium_path


def get_input_data(water_route: str, air_file: str, ammonium_file: str):
	'''
	:param water_route: csv file
	:param air_file: csv file
	:param amoni_file: csv file
	:return: two numpy arrays. The first one with the data (the values) and the
	second one with the boolean columns treated as integers.
	'''
	# Load data
	data_water = pd.read_csv(water_route)
	data_air = pd.read_csv(air_file)
	data_amoni = pd.read_csv(ammonium_file)
	# Join Data
	result = data_water.merge(data_air, on='row_date', how='outer', suffixes=('_air', '_water'))
	result = result.merge(data_amoni, on='row_date').dropna()
	# Split data
	# inputs => The values of water, air and amonium
	inputs = result[['value_water', 'value_air', 'value']]
	# outputs => boolean columns treated as integers
	outputs = result[['is_drift', 'dangerous_drift']].astype(int)
	return inputs.to_numpy(), outputs.to_numpy()


if __name__ == '__main__':
	inputs, outputs = get_input_data(water_path, air_path, ammonium_path)
	model = NEATAlgorithm(neat_config)
	model.fit(inputs, outputs)
