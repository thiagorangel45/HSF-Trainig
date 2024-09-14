import ROOT
import numpy as np
from typing import Optional, Union, List

#-------------------------------
# put a copy of my ROCCurve class
# and other useful functions here
#------------------------------


def branch_to_array(tree: ROOT.TTree,
                    branch_name: str,
                    n_events: Optional[int] = None) -> np.ndarray:
    """!
    @brief Takes ROOT tree and the name of the leaf and returns it as a numpy array

    @param[in] tree: ROOT.TTree object
    @param[in] branch_name: The name of the leaf(branch) you want as a string
    @param[in] n_events: Number of events to use. If not set, defaults to as many as available

    @returns Numpy array of branch
    """
    # each time we'll check to make sure that the event we're on is less than this
    if isinstance(n_events, int):
        maximum_event_number = n_events
    else:
        maximum_event_number = np.inf

    try:  # first try it the fast way
        out = np.array(tree.AsMatrix([branch_name])).flatten()
        if isinstance(maximum_event_number, int):
            out = out[:maximum_event_number]
        return out

    # if that doesn't work due to branches having different names as leaves, try it the slow way
    except:
        out = []
        current_event_number = 0
        for name in get_branch_names(tree):
            if name == branch_name:
                for event in tree:
                    if current_event_number == maximum_event_number:
                        break
                    else:
                        out.append(eval('event.' + branch_name))
                        current_event_number += 1
                return np.array(out)
        else:
            raise ValueError('Branch named "%s" not found in tree %s.' % (branch_name, tree.GetName()))


def get_branch_names(tree: ROOT.TTree) -> List[str]:
    """!
    @brief Get names of branches
    """
    out = []
    for branch in tree.GetListOfBranches():
        out.append(branch.GetName())
    return out


class ROCCurve(object):
    """!
    @brief Describes roc curves
    """

    def __init__(self,
                 mva_targets: Optional[Union[np.ndarray, List[bool]]],
                 scores: Optional[Union[np.ndarray, List[float]]],
                 weights: Optional[Union[np.ndarray, List[float]]] = None):
        """!
        @brief Initialize a receiver operating curve (ROC curve)

        @param[in] mva_targets: Numpy array or Python list of boolean values. True means signal, false background
        @param[in] scores: Score along any range, but generally as floating point values between -1 --> 1 or 0 --> 1
        @param[in] weights: Sometimes TMVA weights different testpoints differently.
        Generally they're all 1 though. Defaults to 1 if not specified
        """
        if mva_targets is not None:
            self.mva_targets = np.array(mva_targets)

        if scores is not None:
            self.scores = np.array(scores)

        if weights is None and (scores is not None):
            self.weights = [1.0] * len(scores)
        else:
            self.weights = weights

        self.roc = None

        self.roc_auc = None

        self.roc_curve = None

        self.roc_graph = None

        self.false_negative_rate = None
        self.true_negative_rate = None

        self.true_positive_rate = None
        self.false_positive_rate = None

    @classmethod
    def from_known_values(cls,
                          signal_efficiency: Union[list, np.ndarray],
                          background_rejection: Union[list, np.ndarray]):
        """!
        @brief Alternate constructor from known signal efficiency array and background rejection

        @param[in] signal_efficiency: list or numpy array of signal efficiency values
        @param[in] background_rejection: list or numpy array of background rejection values
        """
        out = cls(mva_targets=None,
                  scores=None,
                  weights=None)
        out.set_signal_efficiency(signal_efficiency)
        out.set_background_rejection(background_rejection)
        return out

    def get_mva_score_at_sig_acc(self,
                                 target_sig_acc) -> float:
        """!
        @brief Given Background acceptance value, get corresponding BDT score

        @returns single bdt score
        """
        # get signal event scores
        signal_scores = []
        for score, target in zip(self.scores, self.mva_targets):
            if target == True:  # if signal
                signal_scores.append(score)
        signal_scores = np.array(signal_scores)

        # sort them
        signal_scores = np.sort(signal_scores)

        # find what index keeps the correct amount on left and right
        cut_index = int((len(signal_scores)-1) * (1.0 - target_sig_acc))
        val = signal_scores[cut_index]
        return val

    
    def get_mva_score_at_bkg_acc(self,
                                 target_bkg_acc) -> float:
        """!
        @brief Given Background acceptance value, get corresponding BDT score

        @returns single bdt score
        """
        # get background event scores
        background_scores = []
        for score, target in zip(self.scores, self.mva_targets):
            if target == False:  # if background
                background_scores.append(score)
        background_scores = np.array(background_scores)

        # sort them
        background_scores = np.sort(background_scores)

        # find what index keeps the correct amount on left and right
        cut_index = int((len(background_scores)-1) * (1.0 - target_bkg_acc))
        val = background_scores[cut_index]
        return val

    def set_true_positive_rate(self,
                               tpr: Union[list, np.ndarray]) -> None:
        """!
        @brief Set tpr
        @param[in] tpr: list or numpy array of true positive rate values
        @returns None
        """
        self.true_positive_rate = np.array(tpr)

    def set_signal_efficiency(self,
                              signal_efficiency: Union[list, np.ndarray]) -> None:
        """!
        @brief Set tpr

        @param[in] signal_efficiency: list or numpy array of true positive rate values

        @returns None
        """
        self.true_positive_rate = np.array(signal_efficiency)

    def set_sensitivity(self,
                        sensitivity: Union[list, np.ndarray]) -> None:
        """!
        @brief Set tpr

        @param[in] sensitivity: list or numpy array of true positive rate values

        @returns None
        """
        self.true_positive_rate = np.array(sensitivity)

    def set_true_negative_rate(self,
                               tnr: Union[list, np.ndarray]) -> None:
        """!
        @brief Set tnr
        @param[in] tnr: list or numpy array of true negative rate values
        @returns None
        """
        self.true_negative_rate = np.array(tnr)

    def set_specificity(self,
                        specificity: Union[list, np.ndarray]) -> None:
        """!
        @brief Same as set_true_negative_rate

        @param[in] specificity: list or numpy array of true negative rate values

        @returns None
        """
        self.true_negative_rate = np.array(specificity)

    def set_background_rejection(self,
                                 background_rejection: Union[list, np.ndarray]) -> None:
        """!
        @brief Set tnr

        @param[in] background_rejection: list or numpy array of true negative rate values

        @returns None
        """
        self.true_negative_rate = np.array(background_rejection)

    def get_true_positive_rate(self) -> np.ndarray:
        """!
        @brief Get the true positive rate

        @details True positive rate (TPR, sensitivity, signal efficiency).
        \f$ TPR = \frac{TP}{P} = \frac{TP}{TP+FP} = 1 - FNR\f$

        @returns Numpy array
        """
        if self.true_positive_rate is not None:
            return self.true_positive_rate
        else:
            tpr = rootOps.get_tgraph_x(self.get_roc_graph())

            self.true_positive_rate = tpr
            return tpr

    def get_false_negative_rate(self) -> np.ndarray:
        """!
        @brief Get the false negative rate (FNR, miss rate, signal rejection). \f$ FNR = \frac{FN}{P} =
        \frac{FN}{FN+TP} = 1 - TPR \f$

        @returns Numpy array
        """
        if self.false_negative_rate is not None:
            return self.false_negative_rate
        else:
            fnr = 1.0 - self.get_true_positive_rate()
            self.false_negative_rate = fnr
            return fnr

    def get_false_positive_rate(self) -> np.ndarray:
        """!
        @brief Get the false positive rate (FPR, fall-out, background efficiency). \f$ FPR = \frac{FP}{N} =
        \frac{FP}{FP+TN} = 1-TNR \f$

        @returns Numpy array
        """
        if self.false_positive_rate is not None:
            return self.false_positive_rate
        else:
            fpr = 1.0 - self.get_true_negative_rate()

            self.false_positive_rate = fpr
            return fpr

    def get_background_rejection(self) -> np.ndarray:
        """!
        @brief Get background rejection

        @details Returns true negative rate

        @returns Background rejection
        """
        return self.get_true_negative_rate()

    def get_background_efficiency(self) -> np.ndarray:
        """!
        @brief Get background efficiency

        @details Returns false positive rate

        @returns Background efficiency as numpy array
        """
        return self.get_false_positive_rate()

    def get_signal_rejection(self) -> np.ndarray:
        """!
        @brief Get background rejection

        @details Returns false negative rate

        @returns signal rejection as numpy array
        """
        return self.get_false_negative_rate()

    def get_true_negative_rate(self) -> np.ndarray:
        """!
        @brief Get the true negative rate

        @details True negative rate (specificity, TNR, background rejection). \f$ TNR =\frac{TN}{N} =
        \frac{TN}{TN+FP} = 1-FPR \f$

        @returns Numpy array of tnr
        """
        if self.true_negative_rate is not None:
            return self.true_negative_rate
        else:
            tnr = rootOps.get_tgraph_y(self.get_roc_graph())
            self.true_negative_rate = tnr
            return tnr

    def get_specificity(self) -> np.ndarray:
        """!
        @brief Returns true negative rate

        @returns specificity as numpy array
        """
        return self.get_true_negative_rate()

    def get_signal_efficiency(self) -> np.ndarray:
        """!
        @brief Get true positive rate

        @returns signal efficiency as numpy array
        """
        return self.get_true_positive_rate()

    def get_sensitivity(self) -> np.ndarray:
        """!
        @brief Gets true positive rate

        @returns signal efficiency as numpy array
        """
        return self.get_true_positive_rate()

    def get_roc(self) -> ROOT.TMVA.ROCCurve.ROCCurve:
        """!
        @brief Fetches an accurate software-based ROC curve

        @details Creates the ROC curve from scratch by using the training points
        and running them through ROOT::TMVA::ROCCurve::ROCCurve

        @param[in] Testpoints: Testpoints object

        @returns ROOT::TMVA::ROCCurve::ROCCurve object
        """
        if self.roc_curve is not None:
            return self.roc

        else:
            mva_targets = rootOps.array_to_vector(self.mva_targets,
                                                  'Bool_t')
            float_scores = rootOps.array_to_vector(self.scores,
                                                   'Float_t')
            input_points_weights = rootOps.array_to_vector(self.weights,
                                                           'Float_t')
            roc = ROOT.TMVA.ROCCurve(float_scores,  # scores
                                     mva_targets,  # targets
                                     input_points_weights)
            self.roc = roc
            return roc

    def get_roc_auc(self) -> float:
        """!
        @brief Get area under curve of ROC curve

        @returns Area under curve of ROC curve
        """
        if self.roc_auc is not None:
            return self.roc_auc
        else:
            out = ROOT.TMVA.ROCCurve.GetROCIntegral(self.get_roc())
            self.roc_auc = out
            return out

    def get_roc_graph(self) -> ROOT.TGraph:
        """!
        @brief Fetches an accurate software-based ROC curve. True negative rate vs true positive rate

        @details Creates the ROC curve from scratch by using the training points
        and running them through ROOT::TMVA::ROCCurve
        
        @returns ROOT TGraph ROC Curve
        """
        if self.roc_graph is not None:
            return self.roc_graph
        elif (self.true_positive_rate is not None) and (self.true_negative_rate is not None):
            x = self.true_positive_rate
            y = self.true_negative_rate
            n = len(x)

            tg = ROOT.TGraph(n, x, y)
            self.roc_graph = tg
            return tg

        elif (self.mva_targets is not None) and (self.scores is not None) and (self.weights is not None):
            roc = self.get_roc()
            roc_graph = deepcopy(ROOT.TMVA.ROCCurve.GetROCCurve(roc))
            self.roc_graph = roc_graph
            return self.get_roc_graph()

        else:
            raise Exception("You don't know enough values to create a ROC curve.")

    def get_roc_graph_2(self) -> ROOT.TGraph:
        """!
        @brief True positive rate (sensitivity)(signal efficiency)
        vs false positive rate (background efficiency)
       
        @returns TGraph
        """
        x = self.get_false_positive_rate()
        y = self.get_true_positive_rate()
        n = len(x)

        tg = ROOT.TGraph(n, x, y)
        return tg

    def get_roc_graph_3(self) -> ROOT.TGraph:
        """!
        @brief False positive rate (background efficiency)(background acceptance)
        vs True positive rate (sensitivity)(signal efficiency)
        """
        x = self.get_true_positive_rate()
        y = self.get_false_positive_rate()
        n = len(x)

        tg = ROOT.TGraph(n, x, y)
        return tg

    def get_roc_graph_4(self) -> ROOT.TGraph:
        """!
        @brief 1/ False positive rate (1 / background efficiency)
        vs True positive rate (sensitivity)(signal efficiency)
        """
        x1 = self.get_true_positive_rate()
        y1 = self.get_false_positive_rate()

        x = []
        y = []

        for a, b in zip(x1, y1):
            if b != 0:
                x.append(a)
                y.append(1.0 / b)

        n = len(x)
        x = np.array(x)
        y = np.array(y)
        tg = ROOT.TGraph(n, x, y)
        return tg

    def get_roc_graph_5(self) -> ROOT.TGraph:
        """!
        @brief 1 / true negative rate (1 / background rejection)
        vs True positive rate (signal efficiency)
        """
        # start by getting sig eff and bkg rej
        x1 = self.get_true_positive_rate()
        y1 = self.get_true_negative_rate()

        # now get the 1/ values, but only when it wont
        # cause a divide by zero error
        x = []
        y = []

        for a, b in zip(x1, y1):
            if b != 0:
                x.append(a)
                y.append(1.0 / b)

        n = len(x)
        x = np.array(x)
        y = np.array(y)
        tg = ROOT.TGraph(n, x, y)

        return tg

    def get_roc_graph_6(self) -> ROOT.TGraph:
        """!
        @brief 1 / true negative rate (1 / background rejection)
        vs false negative rate (signal efficiency)
        """
        x = self.get_false_negative_rate()
        y = 1.0 / np.array(self.get_true_negative_rate())
        n = len(x)

        tg = ROOT.TGraph(n, x, y)
        return tg
"""
#-----------------------
# train classifiers
#-----------------------
input_file = ROOT.TFile('../../root-tutorial/homework/calorimetry.root', 'READ')  # read in the data file
signal_tree = input_file.Get('eplus')  # identify the signal tree
background_tree = input_file.Get('gamma')  # identify the background tree
output_file = ROOT.TFile('homework-output.root', "RECREATE")  # create an output file for the training results
factory = ROOT.TMVA.Factory('TMVAClassification', output_file)  # setup a TMVA factory

dataloader = ROOT.TMVA.DataLoader('dataset-homework')  # name our dataset 'dataset' because I'm uncreative
# add floating point x and y variables
dataloader.AddVariable('E', 'F')
dataloader.AddVariable('E0', 'F')
dataloader.AddVariable('E1', 'F')
dataloader.AddVariable('E2', 'F')
dataloader.AddVariable('E0frac', 'F')
dataloader.AddVariable('E1frac', 'F')
dataloader.AddVariable('E2frac', 'F')
dataloader.AddVariable('lateralDepth', 'F')
dataloader.AddVariable('lateralDepth2', 'F')
dataloader.AddVariable('showerDepth', 'F')
dataloader.AddVariable('showerDepthWidth', 'F')
dataloader.AddVariable('lateralWidth0', 'F')
dataloader.AddVariable('lateralWidth1', 'F')
dataloader.AddVariable('lateralWidth2', 'F')

# add signal and background trees
dataloader.AddSignalTree(signal_tree)
dataloader.AddBackgroundTree(background_tree)

# set up the training and testing datapoints
cut = ROOT.TCut('')
dataloader.PrepareTrainingAndTestTree(cut, 'SplitMode=Random:NormMode=NumEvents')

# train 3 BDTs
factory.BookMethod(dataloader,
                   ROOT.TMVA.Types.kBDT,
                   'BDT1',
                   'nTrees=100:maxDepth=4:BoostType=AdaBoost')
factory.BookMethod(dataloader,
                   ROOT.TMVA.Types.kBDT,
                   'BDT2',
                   'nTrees=50:maxDepth=8:BoostType=AdaBoost')
factory.BookMethod(dataloader,
                   ROOT.TMVA.Types.kBDT,
                   'BDT3',
                   'nTrees=400:maxDepth=3:BoostType=Grad')
factory.BookMethod(dataloader,
                   ROOT.TMVA.Types.kFisher,
                   'Fisher',
                   'VarTransform=None')
factory.BookMethod(dataloader,
                   ROOT.TMVA.Types.kKNN,
                   'KNN',
                   'nKNN=30')
factory.BookMethod(dataloader,
                   ROOT.TMVA.Types.kCuts,
                   'Cuts',
                   'FitMethod=SA')



# let TMVA do its magic
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
output_file.Close()

"""
#---------------------------------
# plot stuff
#---------------------------------
# get some relevant values
f = ROOT.TFile('homework-output.root')
t = f.Get('dataset-homework/TestTree')

E0_vals = branch_to_array(t, 'E0')
BDT1_scores = branch_to_array(t, 'BDT1')
BDT2_scores = branch_to_array(t, 'BDT2')
BDT3_scores = branch_to_array(t, 'BDT3')
Fisher_scores = branch_to_array(t, 'Fisher')
KNN_scores = branch_to_array(t, 'KNN')
Cuts_scores = branch_to_array(t, 'Cuts')
truth_tags = np.invert(branch_to_array(t, 'classID').astype(bool))

f.Close(); del f

# make some ROC curves
BDT1_roc = ROCCurve(truth_tags, BDT1_scores)
BDT2_roc = ROCCurve(truth_tags, BDT2_scores)
BDT3_roc = ROCCurve(truth_tags, BDT3_scores)
Fisher_roc = ROCCurve(truth_tags, Fisher_scores)
KNN_roc = ROCCurve(truth_tags, KNN_scores)
Cuts_roc = ROCCurve(truth_tags, Cuts_scores)


# get thresholds
background_acceptance = 0.3
BDT1_threshold = BDT1_roc.get_mva_score_at_bkg_acc(background_acceptance)
BDT2_threshold = BDT2_roc.get_mva_score_at_bkg_acc(background_acceptance)
BDT3_threshold = BDT3_roc.get_mva_score_at_bkg_acc(background_acceptance)
Fisher_threshold = Fisher_roc.get_mva_score_at_bkg_acc(background_acceptance)
KNN_threshold = KNN_roc.get_mva_score_at_bkg_acc(background_acceptance)
Cuts_threshold = Cuts_roc.get_mva_score_at_bkg_acc(background_acceptance)

# make tefficiencies
BDT1_eff = ROOT.TEfficiency("BDT1_eff", "BDT1_eff",
                            20, 0, 30000)
BDT2_eff = ROOT.TEfficiency("BDT2_eff", "BDT2_eff",
                            20, 0, 30000)
BDT3_eff = ROOT.TEfficiency("BDT3_eff", "BDT3_eff",
                            20, 0, 30000)
Fisher_eff = ROOT.TEfficiency("Fisher_eff", "Fisher_eff",
                            20, 0, 30000)
KNN_eff = ROOT.TEfficiency("KNN_eff", "KNN_eff",
                            20, 0, 30000)
Cuts_eff = ROOT.TEfficiency("Cuts_eff", "Cuts_eff",
                            20, 0, 30000)


for e0, b1, b2, b3, f, k, c, truth in zip(E0_vals,
                                          BDT1_scores,
                                          BDT2_scores,
                                          BDT3_scores,
                                          Fisher_scores,
                                          KNN_scores,
                                          Cuts_scores,
                                          truth_tags):
    if truth:
        BDT1_eff.Fill(b1 > BDT1_threshold, e0)
        BDT2_eff.Fill(b2 > BDT2_threshold, e0)
        BDT3_eff.Fill(b3 > BDT3_threshold, e0)
        Fisher_eff.Fill(f > Fisher_threshold, e0)
        KNN_eff.Fill(k > KNN_threshold, e0)
        Cuts_eff.Fill(c > Cuts_threshold, e0)
    

# set draw options


# create legend
eff_leg = ROOT.TLegend()
eff_leg.AddEntry(BDT1_eff, "BDT 1", "p")
eff_leg.AddEntry(BDT2_eff, "BDT 2", "p")
eff_leg.AddEntry(BDT3_eff, "BDT 3", "p")
eff_leg.AddEntry(Fisher_eff, "Fisher", "p")
eff_leg.AddEntry(KNN_eff, "KNN", "p")
eff_leg.AddEntry(Cuts_eff, "Cuts", "p")


# plot
eff_canvas = ROOT.TCanvas()
BDT1_eff.Draw()
BDT2_eff.Draw("same")
BDT3_eff.Draw("same")
Fisher_eff.Draw("same")
KNN_eff.Draw("same")
Cuts_eff.Draw("same")
eff_leg.Draw("same")
eff_canvas.Update()
eff_canvas.SaveAs("homework-efficiency-plots.png")
