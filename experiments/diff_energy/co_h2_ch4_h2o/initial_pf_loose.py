"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: 25d1f2d96d73df0774873712e8293fcda3af79f9ee3094bd2f77a6c2e7f46231
Calibration SHA-256: 8a06cb34ab163f1f6e77396200a75b0f49adfdef652d5e3ce5792bcff57e3dba
Calibration SHA-256: 3360c6bc1a6241cd2a2f8d37592f97f22863fbe9f9b63322313ab3c0f34ef7d1
Calibration SHA-256: 5a65aa9c2633bed5369897c9505f1b43f5a4f78d7c0c5d87996333923f6797c4
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed ch4": (137, 7.104453451761818e-13),
        "Reaction seed co": (129, 1.3842976694149891e-12),
        "Reaction seed h2": (109, 1.602852109165564e-10),
        "Reaction seed h2o": (141, 1.7923305397202115e-12),
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
