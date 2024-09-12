import ROOT
import numpy as np
from scipy.stats import rv_continuous
from tqdm import tqdm
from matplotlib import pyplot as plt

#--------------------------------
# define useful function
#--------------------------------
def normalize(hist):
    integral = 0.0
    for i in range(hist.GetNbinsX()+1):
        integral += hist.GetBinContent(i) * hist.GetBinWidth(i)
    hist.Scale(1.0 / integral)
    return hist


# -------------------------------
# setup pathological function
# -------------------------------
class pathological_pdf(rv_continuous):
    def _pdf(self, x, a, b, c):
        return (-a*((x-c)**3) + b) / 72.6495

#--------------------------------
# produce data
# -------------------------------

# true vals
a_true = 0.1
b_true = 9.0
c_true = 4

n_points = 100000
x_min = 0
x_max = 8.481

# create data
my_pdf = pathological_pdf(a=x_min, b=x_max,
                          shapes='a,b,c')
print("Creating data...")
vals = my_pdf.rvs(a_true,
                  b_true,
                  c_true,
                  size=n_points)

np.savetxt('homework-dataset.csv', vals)
print("Data created")