import pandas as pd
from .algorithm.NEATAlgorithm import NEATAlgorithm
from .config import neat_config

def get_input_data(file1, file2, file3):
    # Load data
    data_water = pd.read_csv(file1)
    data_air = pd.read_csv(file2)
    data_amoni = pd.read_csv(file3)
    # Join Data
    result = data_water.merge(data_air, on='row_date', how='outer', suffixes=('_air', '_water'))
    result = result.merge(data_amoni, on='row_date').dropna()
    # Split data
    inputs = result[['value_water', 'value_air', 'value']]
    input_labels = result[['is_drift', 'dangerous_drift']].astype(int)
    return inputs.to_numpy(), input_labels.to_numpy()

model = NEATAlgorithm(neat_config)



