import math
import numpy as np
from hsopt.evaluation.gpu_spectrum import phase_enclosure


def test_actual_compiled_global_phases_are_bounded():
    gates = [
        np.diag(np.exp(1j * np.array([-0.3, 0.3]))) * np.exp(0.05j),
        np.diag(np.exp(1j * np.array([-0.2, 0.2]))) * np.exp(-0.03j),
    ]
    bound = phase_enclosure(gates, 2)
    assert abs(bound["center_radians"] - 0.02) < 1e-12
    assert bound["radius_radians"] >= 0.5
    assert bound["sin_monotone"] and bound["no_repetition_wrap"]
    phases = np.angle(np.linalg.eigvals(gates[1] @ gates[0]))
    assert max(abs(phases - bound["center_radians"])) <= bound["radius_radians"]


def test_large_repetition_aliasing_prevents_lanczos_selection():
    bound = phase_enclosure([np.diag(np.exp(1j * np.array([-0.2, 0.2])))], 20)
    assert bound["sin_monotone"]
    assert not bound["no_repetition_wrap"]


def test_arc_crossing_pi_retains_scalar_phase():
    bound = phase_enclosure([np.diag(np.exp(1j * np.array([math.pi - 0.1, -math.pi + 0.1])))], 1)
    assert abs(abs(bound["center_radians"]) - math.pi) < 1e-12
    assert not bound["sin_monotone"]
