from openevolve import run_evolution, evolve_function

def benchmark_fib(program_path: str) -> float:
    import importlib.util
    import time

    spec = importlib.util.spec_from_file_location("evolved_module", program_path)
    evolved_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evolved_module)

    fibonacci = evolved_module.fibonacci

    test_values = [10, 15, 20]
    expected_results = [55, 610, 6765]

    total_time = 0.0
    for n, expected in zip(test_values, expected_results):
        start_time = time.time()
        result = fibonacci(n)
        end_time = time.time()

        if result != expected:
            return float('inf')  # Penalize incorrect results heavily

        total_time += (end_time - start_time)

    return total_time

# Evolution with inline code (no files needed!)
result = run_evolution(
    initial_program='''
    def fibonacci(n):
        if n <= 1: return n
        return fibonacci(n-1) + fibonacci(n-2)
    ''',
    evaluator=lambda path: {"score": benchmark_fib(path)},
    iterations=100
)

# Evolve Python functions directly
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j] 
    return arr

result = evolve_function(
    bubble_sort,
    test_cases=[([3,1,2], [1,2,3]), ([5,2,8], [2,5,8])],
    iterations=50
)
print(f"Evolved sorting algorithm: {result.best_code}")