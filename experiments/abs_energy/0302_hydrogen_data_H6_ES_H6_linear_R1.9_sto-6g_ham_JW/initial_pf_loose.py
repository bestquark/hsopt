"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: 930141631d7c2a51a98b98f5c650f151f0beaa51b71c2a291623a50c33e6f00a
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "HamLib 0302_hydrogen_data_H6_ES_H6_linear_R1.9_sto-6g_ham_JW": (76, 2.079546050150263e-11),
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
