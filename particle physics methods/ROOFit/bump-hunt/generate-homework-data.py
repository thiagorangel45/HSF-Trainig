import numpy as np

#--------------------------------
# simulate some data
#--------------------------------
background_data = np.random.exponential(scale=40, size=95000)
signal_data = np.random.normal(loc=130, scale=12, size=5000)
data_values = np.concatenate((signal_data, background_data))

np.savetxt('homework-dataset.csv', data_values)