"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline variant: tight
Baseline: Reiher et al. LIQUi|>-style numeric-beta/Eq. (34) PF2 (tight), adapted to 1 kcal/mol.
Calibration SHA-256: 3e764320360bbc9dd2d479deaaa16f8470ac0e7995999ee7e76f8ccb0b1746a3
Full beta/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "HamLib 0277_hydrogen_data_H4_ES_H4_linear_R1.0_sto-6g_ham_JW": (1, 1.926942349220349e-08),
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
