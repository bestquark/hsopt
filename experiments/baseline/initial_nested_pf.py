"""Reiher-style orbital-disjoint nesting, symmetric splitting, uniform synthesis.

This is the starting protocol, not a claim that HSOpt has learned an ELOTE yet.
The host supplies verified fermionic blocks and sealed (r, epsilon, beta) rows.
"""


class ProductFormula:
    hamiltonian_representation = "fermionic_jw"
    baseline_parameters = {}

    def __init__(self, problem):
        self.r, self.epsilon, self.beta = self.baseline_parameters[problem["name"]]
        self.blocks = problem["fermionic_decomposition"]["blocks"]

    def repetitions(self):
        return self.r

    def basis_change(self):
        return []

    def groups(self, terms, step_time):
        # Greedily nest Hermitian terms acting on disjoint spin orbitals.
        layers, occupied = [], []
        lookup = {term["original_label"]: term for term in terms}
        for block in self.blocks:
            layer = 0
            while layer < len(layers) and any(q in occupied[layer] for q in block["orbitals"]):
                layer += 1
            if layer == len(layers):
                layers.append([])
                occupied.append([])
            occupied[layer] += block["orbitals"]
            for entry in block["terms"]:
                term = lookup[entry["label"]]
                layers[layer].append((term, entry["coefficient"] * term["frame_sign"] * step_time))
        return layers

    def build_step(self, terms, step_time):
        groups = self.groups(terms, step_time)
        half = [(index, 0.5) for index in range(len(groups) - 1)]
        return {
            "groups": groups,
            "stages": half + [(len(groups) - 1, 1.0)] + list(reversed(half)),
            "merge_adjacent": True,
        }

    def synthesis_epsilon(self, term, angle):
        return self.epsilon
