"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline variant: tight
Baseline: Reiher et al. LIQUi|>-style numeric-beta/Eq. (34) PF2 (tight), adapted to 1 kcal/mol.
Calibration SHA-256: 27b6ba0028a689b273bafd6bde5735cb7d4a666964844cc760df357ed09a7823
Calibration SHA-256: a8760d8c56100b1ccbd347844d774726d1eba86901269f9b507cec00797c325b
Calibration SHA-256: 8a044f086cff89ec242f5e788fa9b568c1d3f185c07911a5b000882ec4bdc007
Full beta/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed h2": (7, 3.799929883529669e-09),
        "Reaction seed n2": (7, 4.577447236215399e-11),
        "Reaction seed nh3": (10, 1.266828528368643e-11),
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
