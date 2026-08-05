"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline variant: tight
Baseline: Reiher et al. LIQUi|>-style numeric-beta/Eq. (34) PF2 (tight), adapted to 1 kcal/mol.
Calibration SHA-256: 23021cccbe6eb6d9a89226c68fe5fce9fe91b2159f48c42a9098dc659dc29fed
Calibration SHA-256: 6ec4e05d61423d6e3216326c4159b4a82307451a8817dec84b341823027e3231
Calibration SHA-256: 27b6ba0028a689b273bafd6bde5735cb7d4a666964844cc760df357ed09a7823
Calibration SHA-256: 8564f74918b82f5f163809429f381247a16b9f58efae8fbb8409816733057db7
Full beta/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed co": (8, 2.4814095257984293e-11),
        "Reaction seed co2": (5, 1.1216670057669797e-10),
        "Reaction seed h2": (5, 1.2459275178184423e-08),
        "Reaction seed h2o": (13, 2.1243099742538175e-11),
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
