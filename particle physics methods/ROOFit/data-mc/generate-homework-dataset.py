import numpy as np
import ROOT

#-------------------------------------------
# simulate data/mc samples
#-------------------------------------------
# get monte carlo samples
mc_1 = np.random.rayleigh(scale=220, size=3000)
mc_2 = np.random.chisquare(4, size=5000)*50

data = np.concatenate((np.random.rayleigh(scale=220, size=9300),
                       np.random.chisquare(4, size=8600)*50)
                      )
np.savetxt('homework-mc1.csv', mc_1)
np.savetxt('homework-mc2.csv', mc_2)
np.savetxt('homework-data.csv', data)

import matplotlib.pyplot as plt
plt.hist(data, bins=100)
plt.show()