"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: a9783c160976f24813dfafda9525cb4f04f6d940f5ba70a7d2eebd4e14cab1b1
Calibration SHA-256: 9fa11efc4848e228ed1648d0f33f7abd222ddaa250b38a6a291b91c362cf1982
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed hco_plus": (90, 3.678314843687977e-12),
        "Reaction seed hoc_plus": (88, 4.0696297428277356e-12),
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
