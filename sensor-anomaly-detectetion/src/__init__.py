import pandas as pd
import matplotlib.pyplot as plt

def merge_files(file1, file2, file3):
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)
    data3 = pd.read_csv(file3)
    output1 = pd.merge(data1, data2, on='row_date', how='outer', suffixes=('_air', '_water'))
    output = pd.merge(output1, data3, on='row_date')
    output = output[(~output['value_water'].isnull()) & (~output['value_air'].isnull())]

    return output


file_water = '/Users/malal/Desktop/HackEPS/data/aigua.csv'
file_air = '/Users/malal/Desktop/HackEPS/data/aire.csv'
file_amoni = '/Users/malal/Desktop/HackEPS/data/amoni.csv'
out1 = merge_files(file_air, file_water, file_amoni)
print(out1)


