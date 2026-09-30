package pruner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Deterministic cross-layer structured-pruning budget searcher for small
 * feed-forward residual networks.
 *
 * <p>Each scheme selects exactly one pruning candidate per layer and is
 * scored by genuinely executing the fully composed network on the calibration
 * data (see {@link Scorer}); per-layer local errors are never summed.
 *
 * <p>Search strategy with a fixed seed:
 * <ol>
 *   <li><b>Coarse exploration.</b> Candidates are ordered coarsest first
 *       (larger pruning level, then fewer parameters, then original index)
 *       and the coarsest feasible composable scheme anchors the search;
 *       additional schemes are drawn from the coarse end of every layer.</li>
 *   <li><b>Local refinement.</b> Around the current best feasible scheme,
 *       one layer at a time is swapped to neighbouring finer/coarser
 *       candidates, keeping strictly improving moves; occasional seeded
 *       kicks escape local optima.</li>
 *   <li><b>Systematic completion.</b> Any remaining evaluation allowance is
 *       spent enumerating not-yet-evaluated schemes coarsest first.</li>
 * </ol>
 *
 * <p>Evaluation accounting: composing the network and measuring the
 * end-to-end error counts as one evaluation. Schemes that are dimensionally
 * non-composable or exceed the parameter budget are not evaluated. Two
 * selections whose per-layer candidate tensors are identical (duplicate
 * candidates) describe the same composed scheme, share one cached score and
 * are charged only once. On equal error the lexicographically smaller
 * per-layer candidate-index sequence wins, giving a stable result. The best
 * feasible result found so far is always retained; when the evaluation limit
 * is reached the search returns that result normally, and when no feasible
 * scheme exists it reports so explicitly without claiming optimality.
 */
public final class PruningSearcher {

    private final NetworkDefinition network;
    private final CalibrationData data;

    public PruningSearcher(NetworkDefinition network, CalibrationData data) {
        if (network == null) {
            throw new IllegalArgumentException(
                    "network must not be null");
        }
        if (data == null) {
            throw new IllegalArgumentException(
                    "calibration data must not be null");
        }
        int networkInput = network.inputDim();
        int calibrationInput = data.inputs().get(0).length;
        if (networkInput != calibrationInput) {
            throw new IllegalArgumentException(
                    "network input dimension " + networkInput
                            + " does not match calibration input dimension "
                            + calibrationInput);
        }
        int lastLayer = network.layers().size() - 1;
        int networkOutput = network.layers().get(lastLayer).candidates()
                .get(0).outDim();
        int calibrationOutput = data.referenceOutputs().get(0).length;
        if (networkOutput != calibrationOutput) {
            throw new IllegalArgumentException(
                    "network output dimension " + networkOutput
                            + " does not match calibration reference length "
                            + calibrationOutput);
        }
        this.network = network;
        this.data = data;
    }

    public SearchResult search(long budget, long seed, int evaluationLimit) {
        return run(budget, seed, evaluationLimit).result();
    }

    /**
     * Runs one deterministic search and additionally exposes the distinct
     * tensor-content schemes that were actually evaluated, in evaluation
     * order (useful for reproducibility and allowance accounting checks).
     */
    public SearchRun run(long budget, long seed, int evaluationLimit) {
        if (budget < 0) {
            throw new IllegalArgumentException(
                    "budget must not be negative");
        }
        if (evaluationLimit < 0) {
            throw new IllegalArgumentException(
                    "evaluationLimit must not be negative");
        }

        int[][] order = orderedIndices();
        int[][] contentIds = contentIdentities();
        int distinctContentSchemes = distinctContentSchemes(contentIds);
        Random random = new Random(seed);
        Evaluator evaluator = new Evaluator(budget, evaluationLimit,
                contentIds, distinctContentSchemes);

        int[] anchor = firstCandidates(order);
        evaluator.evaluate(anchor);

        runCoarseExploration(order, contentIds, random, evaluator);
        runLocalRefinement(order, contentIds, random, evaluator);
        runSystematicCompletion(order, contentIds, evaluator);

        SearchResult result = evaluator.toResult(seed);
        List<int[]> trace = new ArrayList<>(evaluator.trace.size());
        for (int[] scheme : evaluator.trace) {
            trace.add(scheme.clone());
        }
        return new SearchRun(result, Collections.unmodifiableList(trace));
    }

    /** Result plus the ordered list of schemes actually evaluated. */
    public static final class SearchRun {
        private final SearchResult result;
        private final List<int[]> evaluationTrace;

        private SearchRun(SearchResult result, List<int[]> evaluationTrace) {
            this.result = result;
            this.evaluationTrace = evaluationTrace;
        }

        public SearchResult result() {
            return result;
        }

        public List<int[]> evaluationTrace() {
            return evaluationTrace;
        }
    }

    /**
     * Coarsest-first candidate ordering per layer.
     * Returns original candidate indices in evaluation preference order.
     */
    private int[][] orderedIndices() {
        int layerCount = network.layers().size();
        int[][] order = new int[layerCount][];
        for (int layerIndex = 0; layerIndex < layerCount; layerIndex++) {
            final int layer = layerIndex;
            List<LayerCandidate> candidates =
                    network.layers().get(layer).candidates();
            Integer[] indices = new Integer[candidates.size()];
            for (int candidate = 0; candidate < indices.length;
                    candidate++) {
                indices[candidate] = candidate;
            }
            Arrays.sort(indices, new Comparator<Integer>() {
                @Override
                public int compare(Integer a, Integer b) {
                    LayerCandidate ca =
                            network.layers().get(layer).candidates().get(a);
                    LayerCandidate cb =
                            network.layers().get(layer).candidates().get(b);
                    if (ca.level() != cb.level()) {
                        return Integer.compare(cb.level(), ca.level());
                    }
                    if (ca.retainedParams() != cb.retainedParams()) {
                        return Long.compare(ca.retainedParams(),
                                cb.retainedParams());
                    }
                    return Integer.compare(a, b);
                }
            });
            order[layer] = new int[indices.length];
            for (int candidate = 0; candidate < indices.length;
                    candidate++) {
                order[layer][candidate] = indices[candidate];
            }
        }
        return order;
    }

    private int[] firstCandidates(int[][] order) {
        int[] anchor = new int[order.length];
        for (int layer = 0; layer < order.length; layer++) {
            anchor[layer] = order[layer][0];
        }
        return anchor;
    }

    /**
     * Groups tensor-identical candidates of each layer so that duplicate
     * candidates cannot be charged twice. Each entry maps an original
     * candidate index to a stable per-layer content identity.
     */
    private int[][] contentIdentities() {
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
                if (representative == -1) {
                    ids[candidate] = nextId++;
                } else {
                    ids[candidate] = ids[representative];
                }
            }
            contentIds[layer] = ids;
        }
        return contentIds;
    }

    private static final class CacheKey {
        private final int[] ids;

        CacheKey(int[] ids) {
            this.ids = ids;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof CacheKey)) {
                return false;
            }
            return Arrays.equals(ids, ((CacheKey) other).ids);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(ids);
        }
    }

    /**
     * Charges evaluations, caches scores by tensor content and keeps the
     * stable best feasible scheme found so far.
     */
    private final class Evaluator {
        private final long budget;
        private final int evaluationLimit;
        private final int[][] contentIds;
        private final int distinctContentSchemes;
        private final Scorer scorer = new Scorer(data);
        private final Map<CacheKey, CachedScore> cache = new HashMap<>();
        private final List<int[]> trace = new ArrayList<>();
        private int evaluations;
        private int[] bestSelection;
        private double bestError;
        private long bestParams;

        private Evaluator(long budget, int evaluationLimit,
                          int[][] contentIds, int distinctContentSchemes) {
            this.budget = budget;
            this.evaluationLimit = evaluationLimit;
            this.contentIds = contentIds;
            this.distinctContentSchemes = distinctContentSchemes;
        }

        private boolean allContentSchemesSeen() {
            return distinctContentSchemes >= 0
                    && cache.size() >= distinctContentSchemes;
        }

        private boolean limitReached() {
            return evaluations >= evaluationLimit;
        }

        /**
         * Evaluates one raw candidate-index selection.
         *
         * @return the end-to-end error for an evaluated feasible scheme,
         *         {@code null} when the scheme is non-composable, over
         *         budget, an already-cached duplicate, or the evaluation
         *         allowance is exhausted
         */
        private Double evaluate(int[] selection) {
            if (limitReached()) {
                return null;
            }
            CacheKey key = new CacheKey(toContentKey(selection));
            CachedScore cached = cache.get(key);
            if (cached != null) {
                return cached.feasible ? cached.error : null;
            }

            ComposedNetwork network;
            try {
                network = new ComposedNetwork(
                        PruningSearcher.this.network, selection);
            } catch (CompositionException exception) {
                cache.put(key, CachedScore.INFEASIBLE);
                return null;
            }
            long totalParams = network.totalRetainedParams();
            if (totalParams > budget) {
                cache.put(key, CachedScore.INFEASIBLE);
                return null;
            }

            double error = scorer.meanSquaredError(network);
            evaluations++;
            trace.add(selection.clone());
            cache.put(key, CachedScore.feasible(error));
            if (bestSelection == null || isStrictlyBetter(
                    error, totalParams, selection)) {
                bestSelection = selection.clone();
                bestError = error;
                bestParams = totalParams;
            }
            return error;
        }

        private boolean isStrictlyBetter(double error, long totalParams,
                                         int[] selection) {
            if (error < bestError) {
                return true;
            }
            if (error > bestError) {
                return false;
            }
            return lexicographicallySmaller(selection, bestSelection);
        }

        private int[] toContentKey(int[] selection) {
            int[] key = new int[selection.length];
            for (int layer = 0; layer < selection.length; layer++) {
                key[layer] = contentIds[layer][selection[layer]];
            }
            return key;
        }

        private SearchResult toResult(long seed) {
            boolean limitReached = limitReached();
            if (bestSelection == null) {
                return SearchResult.noFeasible(evaluations, evaluationLimit,
                        budget, limitReached, seed);
            }
            return new SearchResult(true, bestSelection, bestParams,
                    bestError, evaluations, evaluationLimit, budget,
                    limitReached, seed);
        }
    }

    private static boolean lexicographicallySmaller(int[] a, int[] b) {
        for (int layer = 0; layer < a.length; layer++) {
            if (a[layer] != b[layer]) {
                return a[layer] < b[layer];
            }
        }
        return false;
    }

    private static final class CachedScore {
        static final CachedScore INFEASIBLE =
                new CachedScore(false, Double.NaN);

        private final boolean feasible;
        private final double error;

        private CachedScore(boolean feasible, double error) {
            this.feasible = feasible;
            this.error = error;
        }

        static CachedScore feasible(double error) {
            return new CachedScore(true, error);
        }
    }

    private void runCoarseExploration(int[][] order, int[][] contentIds,
                                      Random random, Evaluator evaluator) {
        int layerCount = order.length;
        int poolSize = 1;
        int[] coarseCount = new int[layerCount];
        for (int layer = 0; layer < layerCount; layer++) {
            int medianLevel =
                    network.layers().get(layer).candidates()
                            .get(order[layer][order[layer].length / 2])
                            .level();
            int count = 0;
            for (int position = 0;
                    position < order[layer].length;
                    position++) {
                int candidate = order[layer][position];
                if (network.layers().get(layer).candidates().get(candidate)
                        .level() >= medianLevel) {
                    count++;
                }
            }
            coarseCount[layer] = Math.max(1, count);
            poolSize *= coarseCount[layer];
        }

        int target = evaluator.evaluationLimit / 2;
        if (poolSize <= 64) {
            List<int[]> pool = enumerateCoarseSchemes(order, coarseCount);
            Collections.shuffle(pool, random);
            for (int[] scheme : pool) {
                if (evaluator.limitReached()
                        || evaluator.evaluations >= target) {
                    break;
                }
                evaluator.evaluate(scheme);
            }
            return;
        }

        int stagnationLimit = 64 + 8 * totalCandidateCount();
        int idle = 0;
        while (!evaluator.limitReached()
                && evaluator.evaluations < target
                && idle < stagnationLimit) {
            int[] scheme = new int[layerCount];
            for (int layer = 0; layer < layerCount; layer++) {
                int position = random.nextInt(coarseCount[layer]);
                scheme[layer] = order[layer][position];
            }
            Double error = evaluator.evaluate(scheme);
            idle = error == null ? idle + 1 : 0;
        }
    }

    private List<int[]> enumerateCoarseSchemes(int[][] order,
                                               int[] coarseCount) {
        List<int[]> schemes = new ArrayList<>();
        int[] position = new int[order.length];
        while (true) {
            int[] scheme = new int[order.length];
            for (int layer = 0; layer < order.length; layer++) {
                scheme[layer] = order[layer][position[layer]];
            }
            schemes.add(scheme);

            int layer = 0;
            while (layer < order.length) {
                position[layer]++;
                if (position[layer] < coarseCount[layer]) {
                    break;
                }
                position[layer] = 0;
                layer++;
            }
            if (layer == order.length) {
                return schemes;
            }
        }
    }

    private void runLocalRefinement(int[][] order, int[][] contentIds,
                                    Random random, Evaluator evaluator) {
        if (evaluator.evaluationLimit == 0
                || evaluator.bestSelection == null) {
            return;
        }

        int[] positions = positionsOf(order, evaluator.bestSelection);
        int kickBudget = 2 * order.length + 4;
        int kicks = 0;
        int idleKickProposals = 0;
        int totalCombinations = totalCombinations();
        List<int[]> fullSpaceKicks = totalCombinations <= 64
                ? buildShuffledFullSpace(order, random) : null;

        while (!evaluator.limitReached()
                && !evaluator.allContentSchemesSeen()) {
            boolean improved = false;
            for (int layer = 0;
                    layer < order.length && !evaluator.limitReached();
                    layer++) {
                for (int delta = -1; delta <= 1; delta += 2) {
                    int nextPosition = positions[layer] + delta;
                    if (nextPosition < 0
                            || nextPosition >= order[layer].length) {
                        continue;
                    }
                    int[] candidate = evaluator.bestSelection.clone();
                    candidate[layer] = order[layer][nextPosition];
                    Double errorBefore = evaluator.bestError;
                    Double error = evaluator.evaluate(candidate);
                    if (error != null && error < errorBefore) {
                        positions =
                                positionsOf(order, evaluator.bestSelection);
                        improved = true;
                        kicks = 0;
                        break;
                    }
                }
            }
            if (improved) {
                continue;
            }
            if (kicks >= kickBudget) {
                return;
            }
            boolean proposed = proposeKick(order, random, evaluator,
                    fullSpaceKicks);
            if (!proposed) {
                idleKickProposals++;
                if (idleKickProposals > 64 + totalCandidateCount()) {
                    return;
                }
            } else {
                idleKickProposals = 0;
            }
            kicks++;
        }
    }

    private boolean proposeKick(int[][] order, Random random,
                                Evaluator evaluator,
                                List<int[]> fullSpaceKicks) {
        if (fullSpaceKicks != null) {
            while (!fullSpaceKicks.isEmpty()) {
                int[] scheme = fullSpaceKicks.remove(0);
                if (alreadyEvaluated(evaluator, scheme)) {
                    continue;
                }
                evaluator.evaluate(scheme);
                return true;
            }
            return false;
        }

        int layer = random.nextInt(order.length);
        int[] scheme = evaluator.bestSelection == null
                ? firstCandidates(order) : evaluator.bestSelection.clone();
        scheme[layer] = order[layer][random.nextInt(order[layer].length)];
        evaluator.evaluate(scheme);
        return true;
    }

    private boolean alreadyEvaluated(Evaluator evaluator, int[] scheme) {
        int[] key = new int[scheme.length];
        for (int layer = 0; layer < scheme.length; layer++) {
            key[layer] = evaluator.contentIds[layer][scheme[layer]];
        }
        return evaluator.cache.containsKey(new CacheKey(key));
    }

    private List<int[]> buildShuffledFullSpace(int[][] order,
                                               Random random) {
        List<int[]> schemes = new ArrayList<>();
        int[] position = new int[order.length];
        while (true) {
            int[] scheme = new int[order.length];
            for (int layer = 0; layer < order.length; layer++) {
                scheme[layer] = order[layer][position[layer]];
            }
            schemes.add(scheme);

            int layer = 0;
            while (layer < order.length) {
                position[layer]++;
                if (position[layer] < order[layer].length) {
                    break;
                }
                position[layer] = 0;
                layer++;
            }
            if (layer == order.length) {
                break;
            }
        }
        Collections.shuffle(schemes, random);
        return schemes;
    }

    private void runSystematicCompletion(int[][] order, int[][] contentIds,
                                         Evaluator evaluator) {
        int[] position = new int[order.length];
        long proposals = 0L;
        long proposalCap = Math.min(
                (long) Integer.MAX_VALUE,
                1024L + 64L * Math.max(1, evaluator.evaluationLimit));
        while (!evaluator.limitReached() && proposals < proposalCap) {
            int[] scheme = new int[order.length];
            for (int layer = 0; layer < order.length; layer++) {
                scheme[layer] = order[layer][position[layer]];
            }
            evaluator.evaluate(scheme);
            proposals++;
            if (evaluator.allContentSchemesSeen()) {
                return;
            }

            int layer = 0;
            while (layer < order.length) {
                position[layer]++;
                if (position[layer] < order[layer].length) {
                    break;
                }
                position[layer] = 0;
                layer++;
            }
            if (layer == order.length) {
                return;
            }
        }
    }

    /**
     * Number of distinct per-layer tensor-content combinations, or
     * {@code -1} when the product overflows the small-network range.
     */
    private int distinctContentSchemes(int[][] contentIds) {
        long product = 1L;
        for (int[] ids : contentIds) {
            int distinct = 0;
            int maximum = 0;
            for (int id : ids) {
                maximum = Math.max(maximum, id);
            }
            distinct = maximum + 1;
            if (product > 1_000_000L / distinct) {
                return -1;
            }
            product *= distinct;
        }
        return (int) product;
    }

    private int[] positionsOf(int[][] order, int[] selection) {
        int[] positions = new int[order.length];
        for (int layer = 0; layer < order.length; layer++) {
            for (int position = 0;
                    position < order[layer].length;
                    position++) {
                if (order[layer][position] == selection[layer]) {
                    positions[layer] = position;
                    break;
                }
            }
        }
        return positions;
    }

    private int totalCandidateCount() {
        int count = 0;
        for (LayerSpec spec : network.layers()) {
            count += spec.candidates().size();
        }
        return count;
    }

    private int totalCombinations() {
        int combinations = 1;
        for (LayerSpec spec : network.layers()) {
            if (combinations > 64 / Math.max(1, spec.candidates().size())) {
                return 65;
            }
            combinations *= spec.candidates().size();
        }
        return combinations;
    }

}