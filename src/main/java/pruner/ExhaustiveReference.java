package pruner;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Exhaustive reference optimizer for small networks.
 *
 * <p>Enumerates every per-layer candidate combination, executes each
 * composable distinct-tensor scheme at or below the budget exactly once and
 * returns the optimum together with enumeration statistics. Duplicate
 * candidates (identical tensors within a layer) are executed only once.
 * Intended as a ground-truth reference in tests; it deliberately has no
 * evaluation cap and must not be used on large candidate spaces.
 */
public final class ExhaustiveReference {

    private ExhaustiveReference() {
    }

    /** Exhaustive outcome. {@code feasible} is false when nothing fits. */
    public static final class Optimum {
        private final boolean feasible;
        private final int[] selection;
        private final double error;
        private final long totalRetainedParams;
        private final int feasibleEvaluations;
        private final int composableSchemes;
        private final int totalCombinations;

        Optimum(boolean feasible, int[] selection, double error,
                long totalRetainedParams, int feasibleEvaluations,
                int composableSchemes, int totalCombinations) {
            this.feasible = feasible;
            this.selection = selection;
            this.error = error;
            this.totalRetainedParams = totalRetainedParams;
            this.feasibleEvaluations = feasibleEvaluations;
            this.composableSchemes = composableSchemes;
            this.totalCombinations = totalCombinations;
        }

        public boolean isFeasible() {
            return feasible;
        }

        public int[] selection() {
            return selection == null ? null : selection.clone();
        }

        public double error() {
            return error;
        }

        public long totalRetainedParams() {
            return totalRetainedParams;
        }

        /** Distinct feasible tensor schemes actually executed. */
        public int feasibleEvaluations() {
            return feasibleEvaluations;
        }

        /** Raw selections (index tuples) that are dimensionally composable. */
        public int composableSchemes() {
            return composableSchemes;
        }

        /** Raw index combinations including non-composable ones. */
        public int totalCombinations() {
            return totalCombinations;
        }
    }

    private static final class Key {
        private final int[] ids;

        Key(int[] ids) {
            this.ids = ids;
        }

        @Override
        public boolean equals(Object other) {
            return other instanceof Key
                    && Arrays.equals(ids, ((Key) other).ids);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(ids);
        }
    }

    public static Optimum optimize(NetworkDefinition network,
                                   CalibrationData data, long budget) {
        if (budget < 0) {
            throw new IllegalArgumentException(
                    "budget must not be negative");
        }
        Scorer scorer = new Scorer(data);
        int layerCount = network.layers().size();
        int[] candidateCounts = new int[layerCount];
        int totalCombinations = 1;
        for (int layer = 0; layer < layerCount; layer++) {
            candidateCounts[layer] =
                    network.layers().get(layer).candidates().size();
            totalCombinations *= candidateCounts[layer];
        }

        int[][] contentIds = contentIdentities(network);
        Set<Key> executed = new HashSet<>();
        int[] selection = new int[layerCount];
        int[] best = null;
        double bestError = Double.POSITIVE_INFINITY;
        long bestParams = Long.MAX_VALUE;
        int feasibleEvaluations = 0;
        int composableSchemes = 0;

        while (true) {
            ComposedNetwork composed;
            try {
                composed = new ComposedNetwork(network, selection);
            } catch (CompositionException exception) {
                composed = null;
            }
            if (composed != null) {
                composableSchemes++;
                long params = composed.totalRetainedParams();
                Key key = new Key(contentKey(selection, contentIds));
                if (executed.add(key) && params <= budget) {
                    double error = scorer.meanSquaredError(composed);
                    feasibleEvaluations++;
                    if (best == null || error < bestError
                            || (error == bestError
                                    && lexicographicallySmaller(
                                            selection, best))) {
                        best = selection.clone();
                        bestError = error;
                        bestParams = params;
                    }
                }
            }

            int layer = 0;
            while (layer < layerCount) {
                selection[layer]++;
                if (selection[layer] < candidateCounts[layer]) {
                    break;
                }
                selection[layer] = 0;
                layer++;
            }
            if (layer == layerCount) {
                break;
            }
        }

        if (best == null) {
            return new Optimum(false, null, Double.NaN, 0L,
                    feasibleEvaluations, composableSchemes,
                    totalCombinations);
        }
        return new Optimum(true, best, bestError, bestParams,
                feasibleEvaluations, composableSchemes, totalCombinations);
    }

    private static int[] contentKey(int[] selection, int[][] contentIds) {
        int[] key = new int[selection.length];
        for (int layer = 0; layer < selection.length; layer++) {
            key[layer] = contentIds[layer][selection[layer]];
        }
        return key;
    }

    private static int[][] contentIdentities(NetworkDefinition network) {
        int layerCount = network.layers().size();
        int[][] contentIds = new int[layerCount][];
        for (int layer = 0; layer < layerCount; layer++) {
            List<LayerCandidate> candidates =
                    network.layers().get(layer).candidates();
            int[] ids = new int[candidates.size()];
            int nextId = 0;
            for (int candidate = 0; candidate < candidates.size();
                    candidate++) {
                int representative = -1;
                for (int earlier = 0; earlier < candidate; earlier++) {
                    if (candidates.get(candidate)
                            .sameTensorAs(candidates.get(earlier))) {
                        representative = earlier;
                        break;
                    }
                }
                ids[candidate] = representative == -1
                        ? nextId++ : ids[representative];
            }
            contentIds[layer] = ids;
        }
        return contentIds;
    }

    private static boolean lexicographicallySmaller(int[] a, int[] b) {
        for (int layer = 0; layer < a.length; layer++) {
            if (a[layer] != b[layer]) {
                return a[layer] < b[layer];
            }
        }
        return false;
    }
}
