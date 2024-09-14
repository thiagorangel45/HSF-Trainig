# this file demonstrates using pure TMVA code without any of my helper functions
# to train some events
# first it creates a dataset then classifies it with TMVA

import ROOT
from get_dataset import get_dataset

# create the 'sinusoidal' dataset with 10k points
# saves signal and background in ROOT file 'sinusoidal_1.root' with trees 'signal' and 'background'
get_dataset('sinusoidal_1', 10000)

input_file = ROOT.TFile('sinusoidal_1.root', 'READ')  # read in the data file
data_tree = input_file.Get('tree')  # identify the signal tree
output_file = ROOT.TFile('training-output.root', "RECREATE")  # create an output file for the training results
factory = ROOT.TMVA.Factory('TMVARegression', output_file, "AnalysisType=Regression")  # setup a TMVA factory

dataloader = ROOT.TMVA.DataLoader('dataset')  # name our dataset 'dataset' because I'm uncreative
# add floating point x and y variables
dataloader.AddVariable('v0', 'F')
dataloader.AddTarget('target', 'F')

# add signal and background trees
dataloader.AddRegressionTree(data_tree)

# set up the training and testing datapoints
cut = ROOT.TCut('')
dataloader.PrepareTrainingAndTestTree(cut, 'SplitMode=Random:NormMode=NumEvents')

# book the BDT as a method to use
# apply training options here
factory.BookMethod(dataloader,
                   ROOT.TMVA.Types.kBDT,
                   'BDT',
                   'NTrees=100:MaxDepth=4:BoostType=AdaBoost:SeparationType=RegressionVariance')

# let TMVA do its magic
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
output_file.Close()
