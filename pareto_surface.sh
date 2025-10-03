#!/bin/bash

for lam in $(seq 0 0.1 1.0); do
    printf "\n=== Running with OE_LAMBDA=$lam ===\n"
    OE_LAMBDA=$lam python -m code_evolution.openevolve-run \
        code_evolution/initial_program.py \
        code_evolution/evaluator.py \
        --config code_evolution/config.yaml \
        --iterations 5

    out_dir="code_evolution/openevolve_output"
    new_dir="code_evolution/openevolve_output_${lam}"

    if [ -d "$out_dir" ]; then
        rm -rf "$new_dir"
        mv "$out_dir" "$new_dir"
        echo "Saved results to $new_dir"
    else
        echo "⚠️ Output directory missing for OE_LAMBDA=$lam"
    fi
done