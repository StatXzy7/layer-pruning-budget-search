package pruner;

import java.util.Arrays;

/**
 * Outcome of one deterministic pruning-budget search run.
 *
 * <p>{@code feasible} is {@code false} when no composed network at or below
 * the budget was evaluated. {@code evaluationLimitReached} reports whether the
 * evaluation allowance was consumed. A positive result is a heuristic best
 * effort: the searcher never claims global optimality unless the caller has
 * independently verified (for example by exhaustive enumeration) that the
 * whole feasible space was evaluated.
 */
public final class SearchResult {

    private final boolean feasible;
    private final int[] selection;
    private final Long totalRetainedParams;
    private final Double error;
    private final int evaluations;
    private final int evaluationLimit;
    private final long budget;
    private final boolean evaluationLimitReached;
    private final long seed;

    SearchResult(boolean feasible, int[] selection, Long totalRetainedParams,
                 Double error, int evaluations, int evaluationLimit,
                 long budget, boolean evaluationLimitReached, long seed) {
        this.feasible = feasible;
        this.selection = selection == null ? null : selection.clone();
        this.totalRetainedParams = totalRetainedParams;
        this.error = error;
        this.evaluations = evaluations;
        this.evaluationLimit = evaluationLimit;
        this.budget = budget;
        this.evaluationLimitReached = evaluationLimitReached;
        this.seed = seed;
    }

    static SearchResult noFeasible(int evaluations, int evaluationLimit,
                                   long budget, boolean limitReached,
                                   long seed) {
        return new SearchResult(false, null, null, null, evaluations,
                evaluationLimit, budget, limitReached, seed);
    }

    public boolean isFeasible() {
        return feasible;
    }

    public int[] selection() {
        return selection == null ? null : selection.clone();
    }

    public long totalRetainedParams() {
        if (totalRetainedParams == null) {
            throw new IllegalStateException(
                    "no feasible result has no parameter count");
        }
        return totalRetainedParams;
    }

    public double error() {
        if (error == null) {
            throw new IllegalStateException(
                    "no feasible result has no error");
        }
        return error;
    }

    public int evaluations() {
        return evaluations;
    }

    public int evaluationLimit() {
        return evaluationLimit;
    }

    public long budget() {
        return budget;
    }

    public boolean isEvaluationLimitReached() {
        return evaluationLimitReached;
    }

    public long seed() {
        return seed;
    }

    @Override
    public String toString() {
        if (!feasible) {
            return "no feasible composed network within budget " + budget
                    + " after " + evaluations
                    + (evaluations == 1 ? " evaluation" : " evaluations")
                    + (evaluationLimitReached
                            ? " (evaluation limit reached; result is a "
                                    + "heuristic, not a proven global "
                                    + "optimum)"
                            : " (no feasible scheme exists; this is "
                                    + "exhaustive infeasibility for the "
                                    + "evaluated space)");
        }
        return "feasible heuristic result (not claimed to be globally "
                + "optimal): layers=" + Arrays.toString(selection)
                + ", totalRetainedParams=" + totalRetainedParams
                + ", budget=" + budget
                + ", meanSquaredError=" + error
                + ", evaluations=" + evaluations + "/" + evaluationLimit
                + (evaluationLimitReached
                        ? " (evaluation limit reached)"
                        : " (search exhausted its proposals)")
                + ", seed=" + seed;
    }
}
