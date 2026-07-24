"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: 6922794f3b3d91828ffde8824b2acba0de0767e639133d01bf56a2bc6c0d8dda
Calibration SHA-256: f9577437ab9302d538392bbbaac2e2cdefc9025b612643b5a53162e6c8e2d888
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "HamLib H2 linear R=0.7 STO-6G Jordan-Wigner": (41, 2.3245995758812398e-08),
        "HamLib H2 linear R=0.8 STO-6G Jordan-Wigner": (36, 2.6635314238825205e-08),
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
