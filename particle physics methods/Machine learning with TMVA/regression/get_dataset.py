import ROOT
import argparse
from array import array
from toy_datasets import datasets

def get_dataset(input_dataset, npoints):
    try:
        dataset = datasets[input_dataset]

        f = ROOT.TFile(dataset['name'] + '.root', 'RECREATE')
        t = ROOT.TTree('tree', 'tree')

        data = dataset['function'](npoints)

        if dataset['nreg'] == 1:
            v0 = data['reg_vars'][0]
            target = data['target']

        v0_arr = array('f', [0])
        target_arr = array('f', [0])

        t.Branch('v0', v0_arr, 'v0')
        t.Branch('target', target_arr, 'target')

        if dataset['nreg'] == 1:
            for x, y in zip(v0, target):
                v0_arr[0] = x
                target_arr[0] = y
                t.Fill()

        t.Write()
        f.Close()

        # -------------------------
        # visualize

        f = ROOT.TFile(dataset['name'] + '.root', 'READ')
        t = f.tree

        max_x = max(v0)  # maximum x value
        max_y = max(target)  # maximum y value

        min_x = min(v0)  # minimum x value
        min_y = min(target)  # minimum y value

        x_upper = max_x + 0.1 * abs(max_x)
        x_lower = min_x - 0.1 * abs(min_x)

        y_upper = max_y + 0.1 * abs(max_y)
        y_lower = min_y - 0.1 * abs(min_y)

        if dataset['nreg'] == 1:
            h = ROOT.TH2F('h', '',
                          100, x_lower, x_upper,
                          100, y_lower, y_upper)

            t.Draw('target:v0>>h', '', 'goff')
            h.SetTitle(dataset['name'])
            h.GetXaxis().SetTitle('v_{0}')
            h.GetYaxis().SetTitle('Target')
            h.SetStats(False)
            h.SetMarkerColor(ROOT.kBlue)
            h.SetMarkerStyle(21)

        c = ROOT.TCanvas('c', 'c')
        c.SetGrid()
        h.Draw()
        c.Update()
        c.SaveAs(dataset['name'] + '.png')

    except KeyError:
        print(input_dataset, 'is not one of the possible datasets. Check the README for the full list')
