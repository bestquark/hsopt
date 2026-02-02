"""
Demo script showing how to use the trovolut backend system.

This script demonstrates:
- Listing available backends
- Checking backend availability
- Error handling for missing dependencies
- Using backend components
"""

import trovolut as trv


def main():
    print("=" * 70)
    print("Trovolut Backend System Demo")
    print("=" * 70)
    
    # 1. List all backends with their status
    print("\n📋 All Registered Backends:")
    print("-" * 70)
    all_backends = trv.backends.list_all()
    for name, info in all_backends.items():
        status = "✓ Available" if info["available"] else "✗ Not Available"
        platform = info["platform"]
        print(f"  {name:15} {status:20} Platform: {platform}")
        print(f"  {'':15} {info['description']}")
        print()
    
    # 2. List only available backends
    print("\n✅ Available Backends on This System:")
    print("-" * 70)
    available = trv.backends.list_available()
    if available:
        for backend in available:
            print(f"  • {backend}")
    else:
        print("  No backends available")
    print()
    
    # 3. Demonstrate checking individual backends
    print("\n🔍 Checking Individual Backends:")
    print("-" * 70)
    for backend in ["cpu", "gpu", "qpu", "gpu-emulate"]:
        available = trv.backends.is_available(backend)
        status = "✓" if available else "✗"
        print(f"  {status} {backend:15} {'Available' if available else 'Not Available'}")
    print()
    
    # 4. Try to use GPU backend (demonstrates error handling)
    print("\n🚀 Attempting to Load GPU Backend:")
    print("-" * 70)
    try:
        gpu_backend = trv.backends.require('gpu')
        print("  ✓ GPU backend loaded successfully!")
        print(f"  Module: {gpu_backend}")
    except trv.backends.BackendNotAvailableError as e:
        print(f"  ✗ GPU backend not available:")
        print(f"    {e}")
    print()
    
    # 5. CPU backend should always work
    print("\n💻 Loading CPU Backend (always available):")
    print("-" * 70)
    try:
        cpu_backend = trv.backends.require('cpu')
        print("  ✓ CPU backend loaded successfully!")
        
        # Try to get components
        dynamics = trv.backends.get_dynamics('cpu')
        evaluator = trv.backends.get_evaluator('cpu')
        metrics = trv.backends.get_metrics('cpu')
        
        print(f"  • Dynamics:   {dynamics.__class__.__name__}")
        print(f"  • Evaluator:  {evaluator.__class__.__name__}")
        print(f"  • Metrics:    {metrics.__class__.__name__}")
    except Exception as e:
        print(f"  ✗ Error loading CPU backend: {e}")
    print()
    
    # 6. Show installation instructions for missing backends
    print("\n📦 Installation Instructions:")
    print("-" * 70)
    for backend in ["cpu", "gpu", "qpu", "gpu-emulate"]:
        if not trv.backends.is_available(backend):
            info = all_backends[backend]
            if info["platform"] != "all":
                print(f"  {backend}: uv pip install trovolut[{backend}]")
                print(f"            (Requires {info['platform']} platform)")
            else:
                print(f"  {backend}: uv pip install trovolut[{backend}]")
    
    # All backends
    print(f"\n  all:       uv pip install trovolut[all]")
    print(f"            (Installs all available backends)")
    print()


if __name__ == "__main__":
    main()
