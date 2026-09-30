package prune;

/**
 * Outcome of one seeded budget search.
 *
 * <p>Status semantics:
 * <ul>
 *   <li>{@code INFEASIBLE} - even the cheapest combination exceeds the budget;
 *       no evaluation was performed.</li>
 *   <li>{@code PROVEN_OPTIMAL} - every feasible combination was actually
 *       evaluated, so the result is the global optimum.</li>
 *   <li>{@code CAP_REACHED} - the evaluation budget was exhausted while
 *       feasible combinations remain unevaluated; the result is the best
 *       feasible heuristic result found so far, not a proven optimum.</li>
 * </ul>
 */
public final class SearchResult {
    public enum Status {
        INFEASIBLE,
        PROVEN_OPTIMAL,
        CAP_REACHED
    }

    private final Status status;
    private final int[] selectedCandidateIndices;
    private final int[] selectedLevels;
    private final long totalParams;
    private final double error;
    private final int evaluations;
    private final int feasibleCombinations;
    private final int evaluationCap;
    private final long seed;

    SearchResult(Status status,
                 int[] selectedCandidateIndices,
                 int[] selectedLevels,
                 long totalParams,
                 double error,
                 int evaluations,
                 int feasibleCombinations,
                 int evaluationCap,
                 long seed) {
        this.status = status;
        this.selectedCandidateIndices = selectedCandidateIndices == null
                ? null : selectedCandidateIndices.clone();
        this.selectedLevels = selectedLevels == null ? null : selectedLevels.clone();
        this.totalParams = totalParams;
        this.error = error;
        this.evaluations = evaluations;
        this.feasibleCombinations = feasibleCombinations;
        this.evaluationCap = evaluationCap;
        this.seed = seed;
    }

    public Status status() {
        return status;
    }

    public boolean hasFeasibleResult() {
        return status != Status.INFEASIBLE;
    }

    /** Per-layer index into {@code problem.candidates(layer)}. */
    public int[] selectedCandidateIndices() {
        return selectedCandidateIndices == null ? null : selectedCandidateIndices.clone();
    }

    public int[] selectedLevels() {
        return selectedLevels == null ? null : selectedLevels.clone();
    }

    public long totalParams() {
        return totalParams;
    }

    public double error() {
        return error;
    }

    public int evaluations() {
        return evaluations;
    }

    public int feasibleCombinations() {
        return feasibleCombinations;
    }

    public int evaluationCap() {
        return evaluationCap;
    }

    public long seed() {
        return seed;
    }

    /** Human-readable report; heuristic results are never labelled optimal. */
    public String report() {
        StringBuilder sb = new StringBuilder();
        sb.append("== Pruning budget search report ==\n");
        sb.append("seed            : ").append(seed).append('\n');
        sb.append("evaluation cap  : ").append(evaluationCap).append('\n');
        sb.append("evaluations used: ").append(evaluations).append('\n');
        if (status == Status.INFEASIBLE) {
            sb.append("status          : INFEASIBLE - no combination fits the parameter budget\n");
            return sb.toString();
        }
        sb.append("feasible combos : ").append(feasibleCombinations).append('\n');
        sb.append("status          : ").append(status);
        if (status == Status.CAP_REACHED) {
            sb.append(" (evaluation cap reached; best heuristic result, NOT proven global optimum)");
        }
        sb.append('\n');
        sb.append("selected by layer (candidate index [pruning level]):\n");
        for (int i = 0; i < selectedCandidateIndices.length; i++) {
            sb.append("  layer ").append(i)
              .append(": candidate ").append(selectedCandidateIndices[i])
              .append(" [level ").append(selectedLevels[i]).append("]\n");
        }
        sb.append("total retained parameters: ").append(totalParams).append('\n');
        sb.append(String.format("calibration MSE         : %.10g%n", error));
        return sb.toString();
    }
}
