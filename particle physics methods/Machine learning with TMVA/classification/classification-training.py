# this file demonstrates using pure TMVA code without any of my helper functions
# to train some events
# first it creates a dataset then classifies it with TMVA

import ROOT
from get_dataset import get_dataset

# create the 'moons' dataset with 10k points
# saves signal and background in ROOT file 'moons.root' with trees 'signal' and 'background'
get_dataset('moons', 10000)


input_file = ROOT.TFile('moons.root', 'READ')  # read in the data file
signal_tree = input_file.Get('signal')  # identify the signal tree
background_tree = input_file.Get('background')  # identify the background tree
output_file = ROOT.TFile('training-output.root', "RECREATE")  # create an output file for the training results
factory = ROOT.TMVA.Factory('TMVAClassification', output_file)  # setup a TMVA factory

dataloader = ROOT.TMVA.DataLoader('dataset')  # name our dataset 'dataset' because I'm uncreative
# add floating point x and y variables
dataloader.AddVariable('x', 'F')
dataloader.AddVariable('y', 'F')

# add signal and background trees
dataloader.AddSignalTree(signal_tree)
dataloader.AddBackgroundTree(background_tree)

# set up the training and testing datapoints
cut = ROOT.TCut('')
dataloader.PrepareTrainingAndTestTree(cut, 'SplitMode=Random:NormMode=NumEvents')

# book the BDT as a method to use
# apply training options here
factory.BookMethod(dataloader,
                   ROOT.TMVA.Types.kBDT,
                   'BDT',
                   'nTrees=100:maxDepth=4:BoostType=AdaBoost')

# let TMVA do its magic
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
output_file.Close()
