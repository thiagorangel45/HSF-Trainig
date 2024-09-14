import ROOT
from array import array


# to be run after classification_training.py

x = array('f', [0.0])
y = array('f', [0.0])

reader = ROOT.TMVA.Reader()
reader.AddVariable('x', x)
reader.AddVariable('y', y)

reader.BookMVA('BDT', 'dataset/weights/TMVAClassification_BDT.weights.xml')

# now when you redefine x and y, and call EvaluateMVA, you get the result
x[0] = 12
y[0] = -1.1
bdt_score = reader.EvaluateMVA('BDT')
print(bdt_score)


# again
x[0] = -0.3
y[0] = 4.2
bdt_score = reader.EvaluateMVA('BDT')
print(bdt_score)
