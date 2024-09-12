import ROOT
import numpy as np

#-------------------------------
# define a useful function
#-------------------------------
def normalize(hist):
    integral = 0.0
    for i in range(hist.GetNbinsX()+1):
        integral += hist.GetBinContent(i) * hist.GetBinWidth(i)
    hist.Scale(1.0 / integral)
    return hist


#--------------------------------
# import dataset
#--------------------------------
vals = np.genfromtxt('/Users/thiagorangel/IC CBPF/HSF-Training/particle physics methods/ROOFit/Custom PDF/homework-dataset.csv')

#---------------------------------
# set up RooFit model
# --------------------------------
x_roo = ROOT.RooRealVar("x","x", min(vals), max(vals))
a_roo = ROOT.RooRealVar("a", "a", 0.1, 0, 1)
b_roo = ROOT.RooRealVar("b", "b", 0.1, 0, 1)
c_roo = ROOT.RooRealVar("c", "c", 1, 0, 5)


roo_pdf = ROOT.RooGenericPdf("roo_pdf",
                             "(-a*pow((x-c),3) + b)", [x_roo, a_roo, b_roo, c_roo])


#----------------------------------
# fit to data
#----------------------------------

# make a RooDataSet from the random gaussian set from earlier
data = ROOT.RooDataSet.from_numpy({'x':vals}, [x_roo])

# fit to data
roo_pdf.fitTo(data, 'MaxCalls=10000')

# print fit parameters
print(a_roo)
print(b_roo)
print(c_roo)


#-------------------------------------
# plot results
#-------------------------------------
#----------------------------------
# plot
#----------------------------------
# create data histogram
hist = ROOT.TH1F("hist", "hist", 30, 0, max(vals))
for val in vals:
    hist.Fill(val)
hist = normalize(hist)
hist.SetStats(0)

# create fit histogram
fit_hist = ROOT.TH1F("fit", "fit", 100, 0, max(vals))
for i in range(100):
    x = fit_hist.GetBinCenter(i)
    val = (-1 * a_roo.getVal() * (x-c_roo.getVal())**3) + b_roo.getVal()

    fit_hist.SetBinContent(i, val)
fit_hist = normalize(fit_hist)

hist.SetLineWidth(3)
fit_hist.SetLineWidth(3)
fit_hist.SetFillStyle(0)
fit_hist.SetLineColor(ROOT.kRed-6)

leg = ROOT.TLegend()
leg.AddEntry(hist, 'Data', 'lep')
leg.AddEntry(fit_hist, 'Fit', 'l')

# plot
canvas = ROOT.TCanvas()
hist.Draw()
fit_hist.Draw("hist, c, same")
leg.Draw('same')
canvas.Update()
canvas.SaveAs("homework-plot.png")