"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: 11225dfbe67f2bc80b18d21efd0fbf5a9d0c2ba62ea0ca35acf0ec70e6159c96
Calibration SHA-256: 646a9dd413f4b541d43fd4f9e6adc22a887b783e6f0682f64ff61b76de6b928c
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with CUDA-Q.
"""


class ProductFormula:
    parameters = {
        "Reaction seed hno": (71, 4.6925355704116804e-12),
        "Reaction seed noh": (59, 5.953289545117956e-12),
    }

    def __init__(self, problem):
        self.n_steps, self.epsilon = self.parameters[str(problem["name"])]

    def repetitions(self):
        return self.n_steps

    def build_step(self, terms, step_time):
        ordered = sorted(terms, key=lambda term: term["label"])
        half = [(term, 0.5 * float(term["coefficient"]) * step_time) for term in ordered[:-1]]
        center = [(ordered[-1], float(ordered[-1]["coefficient"]) * step_time)]
        return half + center + list(reversed(half))

    def synthesis_epsilon(self, term, angle):
        _ = (term, angle)
        return self.epsilon
