import ROOT
import numpy as np

m0 = ROOT.RooRealVar('m0', 'm0', 1.4)
sigma = ROOT.RooRealVar('sigma', 'sigma', 3.0, 1e-6, 1e6)
alpha_l = ROOT.RooRealVar('alpha_l', 'alpha_l', 2.0, 1e-6, 1e6)
alpha_r = ROOT.RooRealVar('alpha_r', 'alpha_r', 1.0, 1e-6, 1e6)
eta_l = ROOT.RooRealVar('eta_l', 'eta_l', 0.3, 1e-6, 1e6)
eta_r = ROOT.RooRealVar('eta_r', 'eta_r', 1.9, 1e-6, 1e6)

x = ROOT.RooRealVar("x", "x", -20, 30)
cb = ROOT.RooCrystalBall("cb", "cb", x, m0, sigma, alpha_l, eta_l, alpha_r, eta_r)
ds = cb.generate(x, 10000)
ds.write('crystal-ball-dataset.txt')