from pathlib import Path
from hsopt.evaluation import gpu_spectrum


def test_lanczos_has_no_full_unitary_diagonalization_path():
    source = Path(gpu_spectrum.__file__).read_text()
    assert "full_spectrum(" not in source
    assert "linalg.eig(unitary" not in source
    assert "full_diagonalization_permitted" in source


def test_cupy_public_array_type_uses_explicit_host_transfer():
    import numpy as np
    from hsopt.evaluation.compiled_unitary import _host_array

    class DeviceArray:
        def get(self):
            return np.array([1.0, 2.0])

        def __array__(self, *args, **kwargs):
            raise TypeError("Implicit conversion forbidden")

    DeviceArray.__module__ = "cupy"
    np.testing.assert_equal(_host_array(DeviceArray()), [1.0, 2.0])


def test_cupynumeric_array_does_not_use_cupy_get():
    import numpy as np
    from hsopt.evaluation.compiled_unitary import _host_array

    class LegateArray:
        def __array__(self, *args, **kwargs):
            return np.array([1.0, 2.0])

    LegateArray.__module__ = "cupynumeric._array.array"
    np.testing.assert_equal(_host_array(LegateArray()), [1.0, 2.0])


def test_gpu_reference_is_partial_lanczos_with_analytic_unitary_phase():
    from hsopt.evaluation import compiled_unitary
    import inspect
    reference = inspect.getsource(gpu_spectrum.reference_eigensolve)
    assert 'eigsh(' in reference
    assert 'linalg.eigh' not in reference
    source = inspect.getsource(compiled_unitary.evaluate_exact_reference)
    assert 'gpu_hamiltonian_lanczos_analytic_unitary_phase' in source
