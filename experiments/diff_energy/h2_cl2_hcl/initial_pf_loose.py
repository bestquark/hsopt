"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: fc4ea17ae011a9004b74e61d27d1f4903427bc394b05f6c828054d0aa81800a4
Calibration SHA-256: 3360c6bc1a6241cd2a2f8d37592f97f22863fbe9f9b63322313ab3c0f34ef7d1
Calibration SHA-256: 12eae93eb23e05d3dabcb4270b89944cb77a5a2dbc410e8b67b1e9abc0b6f266
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed cl2": (90, 3.5387393036320947e-12),
        "Reaction seed h2": (55, 1.3495528625213454e-09),
        "Reaction seed hcl": (135, 1.434689363074778e-12),
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
