"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline variant: tight
Baseline: Reiher et al. LIQUi|>-style numeric-beta/Eq. (34) PF2 (tight), adapted to 1 kcal/mol.
Calibration SHA-256: 5088bcd714006b8b6955928ec44c61dc065e59965011ffd57246b97e642278e0
Full beta/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "HamLib 0289_hydrogen_data_H6_ES_H6_linear_R0.6_sto-6g_ham_JW": (1, 9.857780616573986e-10),
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
