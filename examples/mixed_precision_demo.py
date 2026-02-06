"""
Demo of mixed-precision Trotter methods using the GPU emulator backend.

This script demonstrates how to use the gpu-emulate backend to study
quantized exponentials and mixed-precision quantum dynamics, similar to
the mixed-precision tutorial.
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt


def create_test_hamiltonian(n_qubits: int = 4) -> tuple:
    """
    Create a simple test Hamiltonian for demonstration.
    
    Args:
        n_qubits: Number of qubits
        
    Returns:
        Tuple of (H_terms, psi0, dim)
    """
    dim = 2 ** n_qubits
    
    # Create random Hermitian terms
    np.random.seed(42)
    n_terms = 5
    H_terms = []
    
    for i in range(n_terms):
        # Random Hermitian matrix
        H_rand = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        H_herm = (H_rand + H_rand.conj().T) / 2
        
        # Scale by random coefficient
        coeff = np.random.uniform(0.1, 1.0)
        H_terms.append({
            'coeff': coeff,
            'matrix': coeff * H_herm,
        })
    
    # Initial state (product state |0000...>)
    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1.0
    
    return H_terms, psi0, dim


def demo_precision_sweep():
    """Demonstrate precision sweep: compare different precision levels."""
    print("=" * 70)
    print("Mixed-Precision Trotter Demo: Precision Sweep")
    print("=" * 70)
    
    # Import mixed-precision tools
    from trovolut.backends.gpu_mp_emulator.dynamics import Precision
    from trovolut.backends.emulator.evaluator import EmulatorEvaluator
    
    # Create test Hamiltonian
    H_terms, psi0, dim = create_test_hamiltonian(n_qubits=4)
    print(f"\nTest system: {len(H_terms)} Hamiltonian terms, dimension = {dim}")
    
    # Create evaluator
    evaluator = EmulatorEvaluator(
        hamiltonian_terms=H_terms,
        initial_state=psi0,
        evolution_time=1.0,
    )
    print("Reference solution computed (exact FP64)\n")
    
    # Test different precisions
    precisions = [
        Precision.FP64,
        Precision.FP32,
        Precision.FP16,
        Precision.BF16,
        Precision.FP8_E4M3,
        Precision.FP8_E5M2,
    ]
    
    print("Evaluating different precision levels...")
    print("-" * 70)
    results = evaluator.evaluate_precision_sweep(
        precisions=precisions,
        steps=100,
        trotter_order=2,
    )
    
    # Display results
    print(f"\n{'Precision':<15} {'Fidelity':>12} {'Error':>12} {'Time (s)':>12}")
    print("-" * 70)
    for result in results:
        prec_name = result['precisions'][0]  # All same precision
        fidelity = result['fidelity']
        error = result['error']
        time_s = result['time_seconds']
        print(f"{prec_name:<15} {fidelity:12.6f} {error:12.6e} {time_s:12.6f}")
    
    return results


def demo_step_convergence():
    """Demonstrate convergence with increasing Trotter steps."""
    print("\n\n" + "=" * 70)
    print("Mixed-Precision Trotter Demo: Step Convergence")
    print("=" * 70)
    
    from trovolut.backends.gpu_mp_emulator.dynamics import Precision
    from trovolut.backends.emulator.evaluator import EmulatorEvaluator
    
    # Create test Hamiltonian
    H_terms, psi0, dim = create_test_hamiltonian(n_qubits=4)
    print(f"\nTest system: {len(H_terms)} Hamiltonian terms, dimension = {dim}")
    
    # Create evaluator
    evaluator = EmulatorEvaluator(
        hamiltonian_terms=H_terms,
        initial_state=psi0,
        evolution_time=1.0,
    )
    
    # Test different step counts
    step_counts = [10, 20, 50, 100, 200, 500]
    
    print(f"\nTesting FP32 precision with varying step counts...")
    print("-" * 70)
    results = evaluator.evaluate_step_sweep(
        precision=Precision.FP32,
        step_counts=step_counts,
        trotter_order=2,
    )
    
    # Display results
    print(f"\n{'Steps':>8} {'Fidelity':>12} {'Error':>12} {'Time (s)':>12}")
    print("-" * 70)
    for result in results:
        steps = result['steps']
        fidelity = result['fidelity']
        error = result['error']
        time_s = result['time_seconds']
        print(f"{steps:8d} {fidelity:12.6f} {error:12.6e} {time_s:12.6f}")
    
    return results


def demo_mixed_precision_config():
    """Demonstrate mixed-precision configuration: different precisions per term."""
    print("\n\n" + "=" * 70)
    print("Mixed-Precision Trotter Demo: Mixed-Precision Configurations")
    print("=" * 70)
    
    from trovolut.backends.gpu_mp_emulator.dynamics import Precision
    from trovolut.backends.emulator.evaluator import EmulatorEvaluator
    
    # Create test Hamiltonian
    H_terms, psi0, dim = create_test_hamiltonian(n_qubits=4)
    n_terms = len(H_terms)
    print(f"\nTest system: {n_terms} Hamiltonian terms, dimension = {dim}")
    
    # Create evaluator
    evaluator = EmulatorEvaluator(
        hamiltonian_terms=H_terms,
        initial_state=psi0,
        evolution_time=1.0,
    )
    
    # Define different precision configurations
    configs = [
        [Precision.FP64] * n_terms,
        [Precision.FP32] * n_terms,
        [Precision.FP16] * n_terms,
        # Mixed: important terms in FP32, others in FP16
        [Precision.FP32, Precision.FP16, Precision.FP16, Precision.FP16, Precision.FP16],
        # Mixed: gradient of precisions
        [Precision.FP32, Precision.FP16, Precision.FP16, Precision.BF16, Precision.FP8_E4M3],
    ]
    
    config_names = [
        "All FP64",
        "All FP32",
        "All FP16",
        "Mixed FP32/FP16",
        "Gradient (FP32→FP8)",
    ]
    
    print("\nComparing mixed-precision configurations...")
    print("-" * 70)
    results = evaluator.compare_mixed_precision(
        precision_configs=configs,
        config_names=config_names,
        steps=100,
        trotter_order=2,
    )
    
    # Display results
    print(f"\n{'Configuration':<25} {'Fidelity':>12} {'Error':>12} {'Time (s)':>12}")
    print("-" * 70)
    for result in results:
        name = result['config_name']
        fidelity = result['fidelity']
        error = result['error']
        time_s = result['time_seconds']
        print(f"{name:<25} {fidelity:12.6f} {error:12.6e} {time_s:12.6f}")
    
    return results


def demo_trotter_orders():
    """Demonstrate different Trotter orders."""
    print("\n\n" + "=" * 70)
    print("Mixed-Precision Trotter Demo: Trotter Order Comparison")
    print("=" * 70)
    
    from trovolut.backends.gpu_mp_emulator.dynamics import Precision
    from trovolut.backends.emulator.evaluator import EmulatorEvaluator
    
    # Create test Hamiltonian
    H_terms, psi0, dim = create_test_hamiltonian(n_qubits=4)
    print(f"\nTest system: {len(H_terms)} Hamiltonian terms, dimension = {dim}")
    
    # Create evaluator
    evaluator = EmulatorEvaluator(
        hamiltonian_terms=H_terms,
        initial_state=psi0,
        evolution_time=1.0,
    )
    
    # Test different Trotter orders
    orders = [1, 2, 4]
    
    print("\nComparing Trotter orders (FP32, 50 steps)...")
    print("-" * 70)
    
    results = []
    for order in orders:
        print(f"Evaluating order {order}...")
        result = evaluator.evaluate_single(
            precisions=Precision.FP32,
            steps=50,
            trotter_order=order,
        )
        results.append(result)
    
    # Display results
    print(f"\n{'Order':>8} {'Fidelity':>12} {'Error':>12} {'Time (s)':>12}")
    print("-" * 70)
    for result in results:
        order = result['trotter_order']
        fidelity = result['fidelity']
        error = result['error']
        time_s = result['time_seconds']
        print(f"{order:8d} {fidelity:12.6f} {error:12.6e} {time_s:12.6f}")
    
    return results


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "   Mixed-Precision Trotter Methods Demo   ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("║" + "   GPU Emulator Backend (trovolut)   ".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Check if backend is available
    try:
        import trovolut as trv
        if not trv.backends.is_available('gpu-emulate'):
            print("\n❌ GPU emulator backend not available!")
            print("Install with: uv pip install trovolut[gpu-emulate]")
            return
    except ImportError:
        print("\n❌ Trovolut not installed!")
        print("Install with: uv pip install trovolut")
        return
    
    # Run demonstrations
    try:
        demo_precision_sweep()
        demo_step_convergence()
        demo_mixed_precision_config()
        demo_trotter_orders()
        
        print("\n\n" + "=" * 70)
        print("✅ All demonstrations completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
