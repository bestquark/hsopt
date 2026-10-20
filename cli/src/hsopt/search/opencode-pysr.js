// OpenCode v2 plugin. The host owns the sealed suite, compiler and evaluation budget.
export default {
  id: "hsopt.pysr",
  async setup(ctx) {
    const endpoint = new URL("/search", ctx.options.endpoint).href;
    await ctx.tool.transform((editor) => {
      editor.add({
        name: "hsopt_pysr",
        description:
          "Optimize a mixed-precision synthesis equation with PySR on the sealed training suite. " +
          "Pass the complete proposed ProductFormula as base_policy: its splitting, grouping and " +
          "repetitions stay fixed during this inner search. Features: c=relative coefficient, " +
          "w=relative Pauli support, a=relative rotation angle, d=1 for I/Z-only, " +
          "xy=relative X/Y count, lc=log(c). It learns delta in epsilon0*exp(clamp(delta,-8,16)). " +
          "All probes consume the SAME campaign evaluation budget. Returns certified training " +
          "results and best.source; inspect it, then write your final candidate.py once. " +
          "No test data is accessible. The tool never edits candidate.py.",
        input: {
          type: "object",
          properties: {
            base_policy: { type: "string", description: "Complete proposed policy source." },
            guess: { type: "string", description: "Optional arithmetic starting expression; default 0." },
            max_evals: { type: "integer", minimum: 1, description: "Maximum new suite evaluations; host caps it." },
            seconds: { type: "number", exclusiveMinimum: 0 },
            maxsize: { type: "integer", minimum: 3, maximum: 20 },
            operators: { type: "array", items: { type: "string", enum: ["+", "-", "*", "/", "abs", "sqrt"] } },
          },
          required: ["base_policy"],
          additionalProperties: false,
        },
        async execute(input) {
          const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json", "Authorization": `Bearer ${ctx.options.token}` },
            body: JSON.stringify(input),
            signal: AbortSignal.timeout(86400000),
          });
          const result = await response.json();
          if (!response.ok) throw new Error(result.error || `PySR service HTTP ${response.status}`);
          return { content: JSON.stringify(result) };
        },
      });
    });
  },
};
