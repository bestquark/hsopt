"""Generated iteration-0 product formula; mutate the class, not the benchmark.

Baseline: Reiher et al. Eq. (19)/(34) rigorous PF2, adapted to 1 kcal/mol.
Calibration SHA-256: 2303a70ed784ef28a5e8cefe670e5bcf01c842a806fc96af1a5d80742f10836d
Calibration SHA-256: f4a02811d32ef062fab0760c4d3fb62b94cab0f8dbba19c262ece5ef0ffab5c7
Calibration SHA-256: a7b608a5171005f7b139a885fc3d61967dee9bb86efbcfe37fcd1e4f3258ce52
Full bound/error derivation and the exact-energy reference are sealed outside
this mutation file. Iteration 0 measures this code unchanged with PyGridSynth.
"""


class ProductFormula:
    parameters = {
        "Reaction seed f2": (6139, 2.667910226642987e-12),
        "Reaction seed h2": (411, 1.3703794022127365e-09),
        "Reaction seed hf": (16652, 6.331526289486157e-13),
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
