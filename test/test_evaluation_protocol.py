import unittest

from experiments.evaluation_protocol.evaluation_protocol import ContextFeature, EvaluationProtocol
from experiments.evaluation_protocol.plot_evaluate_on_protocol import plot_evaluation_protocol


class TestEvaluationProtocol(unittest.TestCase):
    def test_context_creation(self):
        cf0 = ContextFeature("g", 9., 9.5, 10., 11.)
        cf1 = ContextFeature("l", 0.4, 0.5, 0.6, 0.8)
        seed = None
        n_contexts = 100
        context_features = [cf0, cf1]
        modes = ["A", "B", "C"]
        for mode in modes:
            ep = EvaluationProtocol(context_features=context_features, mode=mode, seed=seed)
            contexts_train = ep.create_train_contexts(n=n_contexts)
            contexts_ES = ep.create_contexts_extrapolation_single(n=n_contexts)  # covers two quadrants
            contexts_EA = ep.create_contexts_extrapolation_all(n=n_contexts)
            contexts_I = ep.create_contexts_interpolation(n=n_contexts, contexts_forbidden=contexts_train)
            contexts_IC = ep.create_contexts_interpolation_combinatorial(n=n_contexts, contexts_forbidden=contexts_train)
            contexts_dict = {
                "train": contexts_train,
                "test_interpolation": contexts_I,
                "test_interpolation_combinatorial": contexts_IC,
                "test_extrapolation_single": contexts_ES,
                "test_extrapolation_all": contexts_EA,
            }
            for c_id, C in contexts_dict.items():
                if len(C) != 0:
                    self.assertTrue(len(C) == n_contexts, msg=f"Number of contexts {len(C)} not equal to desired number {n_contexts} for {c_id}.")

    def test_plot(self):
        cf0 = ContextFeature("g", 9., 9.5, 10., 11.)
        cf1 = ContextFeature("l", 0.4, 0.5, 0.6, 0.8)
        seed = 1
        n_contexts = 100
        context_features = [cf0, cf1]
        plot_evaluation_protocol(context_features=context_features, seed=seed, n_contexts=n_contexts)
