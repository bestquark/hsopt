"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: e51c84b32b9bf0fff14224dafedecd7d05606e400486b21a6d9cdb3acda0aaba
Calibration SHA-256: 3360c6bc1a6241cd2a2f8d37592f97f22863fbe9f9b63322313ab3c0f34ef7d1
Calibration SHA-256: fd9e84ce6d8d90b2c7b1d30b3c0ecc1b0c8515a79d404d15df60edbd39fc0eab
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with CUDA-Q.
"""


class ProductFormula:
    parameters = {
        "Reaction seed f2": (88, 2.650566360025094e-12),
        "Reaction seed h2": (55, 1.3495528625213454e-09),
        "Reaction seed hf": (181, 6.30777781357471e-13),
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
