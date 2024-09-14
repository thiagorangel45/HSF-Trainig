import ROOT
import numpy as np
from typing import Union, Optional, List, Tuple
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

    def get_signal_events_vs_score(self, n_points: int = 2000) -> Tuple[np.ndarray]:
        """!
        @brief Get signal event counts above each score
        """
        scores = np.linspace(min(self.scores),
                             max(self.scores),
                             n_points)
        signal_event_counts = np.zeros(n_points)
        for t, s, w in zip(self.mva_targets, self.scores, self.weights):
            if t:  # if signal
                idx = arrayOps.find_index_of_nearest(scores, s)
                signal_event_counts[:idx] += w
        return scores, signal_event_counts

    def get_background_events_vs_score(self, n_points: int = 2000) -> Tuple[np.ndarray]:
        """!
        @brief Get background event counts above each score
        """
        scores = np.linspace(min(self.scores),
                             max(self.scores),
                             n_points)
        background_event_counts = np.zeros(n_points)
        for t, s, w in zip(self.mva_targets, self.scores, self.weights):
            if not t:  # if background
                idx = arrayOps.find_index_of_nearest(scores, s)
                background_event_counts[:idx] += w
        return scores, background_event_counts

    def get_sensitivity_vs_score(self, n_points: int=2000) -> Tuple[np.ndarray]:
        """!
        @brief Get sensitivity at each BDT output score cut
        """
        scores, sig_counts = self.get_signal_events_vs_score(n_points=n_points)
        _, bkg_counts =  self.get_background_events_vs_score(n_points=n_points)

        sensitivity_out = []
        scores_out = []
        for a,b,c in zip(scores, sig_counts, bkg_counts):
            if (c!=0):
                sensitivity_out.append(b/np.sqrt(c))
                scores_out.append(a)
        return scores_out, sensitivity_out
        
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
