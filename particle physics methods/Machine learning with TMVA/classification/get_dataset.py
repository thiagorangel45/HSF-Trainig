import ROOT
import argparse
from array import array
from toy_datasets import datasets

def get_dataset(input_dataset, npoints):
    try:
        dataset = datasets[input_dataset]

        f = ROOT.TFile(dataset['name'] + '.root', 'RECREATE')
        sig = ROOT.TTree('signal', 'signal')
        bkg = ROOT.TTree('background', 'background')

        data = dataset['function'](npoints)

        if dataset['nvar'] == 2:
            signal_x, signal_y = data['signal_vars']
            background_x, background_y = data['background_vars']
        elif dataset['nvar'] == 3:
            signal_x, signal_y, signal_z = data['signal_vars']
            background_x, background_y, background_z = data['background_vars']

        x_sig = array('f', [0])
        y_sig = array('f', [0])

        x_bkg = array('f', [0])
        y_bkg = array('f', [0])

        sig.Branch('x', x_sig, 'x')
        sig.Branch('y', y_sig, 'y')
        bkg.Branch('x', x_bkg, 'x')
        bkg.Branch('y', y_bkg, 'y')

        if dataset['nvar'] == 3:
            z_sig = array('f', [0])
            z_bkg = array('f', [0])

            sig.Branch('z', z_sig, 'z'),
            bkg.Branch('z', z_bkg, 'z')

        if dataset['nvar'] == 2:
            for x, y in zip(signal_x, signal_y):
                x_sig[0] = x
                y_sig[0] = y
                sig.Fill()
            for x, y in zip(background_x, background_y):
                x_bkg[0] = x
                y_bkg[0] = y
                bkg.Fill()

        elif dataset['nvar'] == 3:
            for x, y, z in zip(signal_x, signal_y, signal_z):
                x_sig[0] = x
                y_sig[0] = y
                z_sig[0] = z
                sig.Fill()
            for x, y, z in zip(background_x, background_y, background_z):
                x_bkg[0] = x
                y_bkg[0] = y
                z_bkg[0] = z
                bkg.Fill()

        sig.Write()
        bkg.Write()
        f.Close()

        # -------------------------
        # visualize

        f = ROOT.TFile(dataset['name'] + '.root', 'READ')
        sig_tree = f.signal
        bkg_tree = f.background

        max_x = max(signal_x + background_x)  # maximum x value
        max_y = max(signal_y + background_y)  # maximum y value

        min_x = min(signal_x + background_x)  # minimum x value
        min_y = min(signal_y + background_y)  # minimum y value

        x_upper = max_x + 0.1 * abs(max_x)
        x_lower = min_x - 0.1 * abs(min_x)

        y_upper = max_y + 0.1 * abs(max_y)
        y_lower = min_y - 0.1 * abs(min_y)

        if dataset['nvar'] == 3:
            max_z = max(signal_z + background_z)
            min_z = min(signal_z + background_z)

            z_upper = max_z + 0.1 * abs(max_z)
            z_lower = min_z - 0.1 * abs(min_z)

        if dataset['nvar'] == 2:
            axes_hist = ROOT.TH2F('axes_hist', '',
                                  1, x_lower, x_upper,
                                  1, y_lower, y_upper)
            axes_hist.SetTitle(dataset['name'] + ': Signal and Background')
            axes_hist.GetXaxis().SetTitle('x')
            axes_hist.GetYaxis().SetTitle('y')
            axes_hist.SetStats(False)

            leg = ROOT.TLegend()

            signal_hist = ROOT.TH2F('signal_hist', '',
                                    100, x_lower, x_upper,
                                    100, y_lower, y_upper)

            sig_tree.Draw('y:x>>signal_hist', '', 'goff')

            background_hist = ROOT.TH2F('background_hist', '',
                                        100, x_lower, x_upper,
                                        100, y_lower, y_upper)

            bkg_tree.Draw('y:x>>background_hist', '', 'goff')

            signal_hist.SetMarkerColor(ROOT.kYellow - 3)
            signal_hist.SetFillColor(ROOT.kYellow - 3)
            signal_hist.SetStats(False)
            leg.AddEntry(signal_hist, 'signal', 'f')

            background_hist.SetMarkerColor(ROOT.kBlue)
            background_hist.SetFillColor(ROOT.kBlue)
            background_hist.SetStats(False)
            leg.AddEntry(background_hist, 'background', 'f')


        elif dataset['nvar'] == 3:
            axes_hist = ROOT.TH3F('axes_hist', '',
                                  1, x_lower, x_upper,
                                  1, y_lower, y_upper,
                                  1, z_lower, z_upper)
            axes_hist.SetTitle(dataset['name'] + ': Signal and Background')
            axes_hist.GetXaxis().SetTitle('x')
            axes_hist.GetYaxis().SetTitle('y')
            axes_hist.GetZaxis().SetTitle('z')
            axes_hist.SetStats(False)

            leg = ROOT.TLegend()

            signal_hist = ROOT.TH3F('signal_hist', '',
                                    100, x_lower, x_upper,
                                    100, y_lower, y_upper,
                                    100, z_lower, z_upper)

            sig_tree.Draw('z:y:x>>signal_hist', '', 'goff')

            background_hist = ROOT.TH3F('background_hist', '',
                                        100, x_lower, x_upper,
                                        100, y_lower, y_upper,
                                        100, z_lower, z_upper)

            bkg_tree.Draw('z:y:x>>background_hist', '', 'goff')

            signal_hist.SetMarkerColor(ROOT.kYellow - 3)
            signal_hist.SetFillColor(ROOT.kYellow - 3)
            signal_hist.SetStats(False)
            leg.AddEntry(signal_hist, 'signal', 'f')

            background_hist.SetMarkerColor(ROOT.kBlue)
            background_hist.SetFillColor(ROOT.kBlue)
            background_hist.SetStats(False)
            leg.AddEntry(background_hist, 'background', 'f')

        c = ROOT.TCanvas('c', 'c')
        c.SetGrid()
        axes_hist.Draw()
        signal_hist.Draw('SAME')
        background_hist.Draw('SAME')
        leg.Draw('SAME')
        c.Update()
        c.SaveAs(dataset['name'] + '.png')


    except KeyError:
        print(input_dataset, 'is not one of the possible datasets. Check the README for the full list')
