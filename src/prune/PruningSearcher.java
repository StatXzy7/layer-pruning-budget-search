package prune;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Cross-layer structured-pruning budget searcher.
 *
 * <p>Exactly one already-pruned, compensated candidate is chosen per
 * layer; a combination is feasible when the sum of the candidates' true
 * retained parameter counts does not exceed the budget. Every charged
 * evaluation actually composes the chosen matrices into a full network
 * and scores its end-to-end calibration MSE; per-layer local errors are
 * never summed.
 *
 * <p>The seeded, finite search first explores coarser pruning levels
 * (candidates ordered by retained parameters ascending, i.e. most
 * aggressive pruning first), then refines feasible solutions with finer
 * candidates via hill climbing and biased random restarts. When the
 * evaluation cap permits, every remaining feasible combination is
 * enumerated, making the result the proven global optimum; otherwise the
 * best feasible heuristic result is returned as {@code CAP_REACHED} and
 * is never labelled optimal.
 *
 * <p>A combination whose composed matrices equal those of a previously
 * evaluated combination is served from cache and does not consume the
 * evaluation budget, even when assembled from duplicated candidates.
 */
public final class PruningSearcher {
    private static final int RESTARTS = 3;
    private static final int SAMPLE_ATTEMPTS = 64;

    private final PruningProblem problem;
    private final Evaluator evaluator;
    private final int layerCount;
    private final int[] widths;
    private final double[][][][] candidateWeights;
    private final long[][] candidateParams;
    private final int[][] candidateLevels;
    /** Per-layer raw indices ordered coarse (cheap) to fine (expensive). */
    private final int[][] order;
    private final long[] minParams;

    public PruningSearcher(PruningProblem problem, Evaluator evaluator) {
        if (problem == null) {
            throw new InvalidProblemException("problem must not be null");
        }
        if (evaluator == null) {
            throw new InvalidProblemException("evaluator must not be null");
        }
        this.problem = problem;
        this.evaluator = evaluator;
        this.layerCount = problem.layerCount();
        this.widths = new int[layerCount];
        this.candidateWeights = new double[layerCount][][][];
        this.candidateParams = new long[layerCount][];
        this.candidateLevels = new int[layerCount][];
        this.order = new int[layerCount][];
        this.minParams = new long[layerCount];
        init();
    }

    public PruningSearcher(final PruningProblem problem) {
        this(problem, new Evaluator() {
            @Override
            public double evaluate(double[][][] composedWeights) {
                NetworkConfig composed = problem.network().withWeights(composedWeights);
                return NetworkRunner.meanSquaredError(composed,
                        problem.calibrationInputs(), problem.referenceOutputs());
            }
        });
    }

    private void init() {
        for (int l = 0; l < layerCount; l++) {
            final List<Candidate> cands = problem.candidates(l);
            final int n = cands.size();
            widths[l] = n;
            candidateWeights[l] = new double[n][][];
            candidateParams[l] = new long[n];
            candidateLevels[l] = new int[n];
            Integer[] idx = new Integer[n];
            for (int k = 0; k < n; k++) {
                Candidate c = cands.get(k);
                candidateWeights[l][k] = c.weights();
                candidateParams[l][k] = c.retainedParams();
                candidateLevels[l][k] = c.level();
                idx[k] = k;
            }
            final int layer = l;
            Arrays.sort(idx, new Comparator<Integer>() {
                @Override
                public int compare(Integer a, Integer b) {
                    long pa = candidateParams[layer][a];
                    long pb = candidateParams[layer][b];
                    if (pa != pb) {
                        return Long.compare(pa, pb);
                    }
                    int la = candidateLevels[layer][a];
                    int lb = candidateLevels[layer][b];
                    if (la != lb) {
                        return Integer.compare(lb, la); // coarser (higher level) first
                    }
                    return Integer.compare(a, b);
                }
            });
            order[l] = new int[n];
            for (int k = 0; k < n; k++) {
                order[l][k] = idx[k];
            }
            minParams[l] = candidateParams[l][order[l][0]];
        }
    }

    public SearchResult run(long seed, int maxEvaluations) {
        if (maxEvaluations <= 0) {
            throw new IllegalArgumentException("maxEvaluations must be positive, got " + maxEvaluations);
        }
        return new Session(seed, maxEvaluations).run();
    }

    private final class Session {
        private final long seed;
        private final int cap;
        private final Random random;
        private final Map<MatrixKey, Double> cache = new HashMap<MatrixKey, Double>();
        private int evaluations;
        private int feasibleCount;

        private int[] bestTuple;
        private double bestError = Double.POSITIVE_INFINITY;
        private long bestParams = -1L;

        private Session(long seed, int cap) {
            this.seed = seed;
            this.cap = cap;
            this.random = new Random(seed);
        }

        private SearchResult run() {
            long minSum = 0L;
            for (long p : minParams) {
                minSum += p;
            }
            if (minSum > problem.paramBudget()) {
                return new SearchResult(SearchResult.Status.INFEASIBLE, null, null,
                        minSum, Double.POSITIVE_INFINITY, 0, 0, cap, seed);
            }
            feasibleCount = countFeasible();

            // Coarsest feasible combination: every layer at its cheapest candidate.
            int[] start = new int[layerCount];
            evaluate(start);
            hillClimb(start);
            if (capReached()) {
                return finish(false);
            }

            // Biased random restarts around feasible, coarse regions.
            for (int r = 0; r < RESTARTS && !capReached(); r++) {
                int[] t = sampleFeasible();
                if (t == null) {
                    break;
                }
                evaluate(t);
                hillClimb(t);
            }
            if (capReached()) {
                return finish(false);
            }

            // Budget-ordered exhaustive sweep of every feasible combination.
            boolean exhausted = enumerate();
            return finish(exhausted);
        }

        private SearchResult finish(boolean proven) {
            SearchResult.Status status = proven
                    ? SearchResult.Status.PROVEN_OPTIMAL
                    : SearchResult.Status.CAP_REACHED;
            int[] rawIdx = new int[layerCount];
            int[] levels = new int[layerCount];
            for (int l = 0; l < layerCount; l++) {
                int raw = order[l][bestTuple[l]];
                rawIdx[l] = raw;
                levels[l] = candidateLevels[l][raw];
            }
            return new SearchResult(status, rawIdx, levels, bestParams, bestError,
                    evaluations, feasibleCount, cap, seed);
        }

        private boolean capReached() {
            return evaluations >= cap;
        }

        private long paramsOf(int[] tuple) {
            long sum = 0L;
            for (int l = 0; l < layerCount; l++) {
                sum += candidateParams[l][order[l][tuple[l]]];
            }
            return sum;
        }

        private boolean feasible(int[] tuple) {
            return paramsOf(tuple) <= problem.paramBudget();
        }

        /**
         * Returns the end-to-end error of a tuple, charging one evaluation
         * only the first time its composed matrices are seen.
         */
        private double evaluate(int[] tuple) {
            double[][][] composed = new double[layerCount][][];
            for (int l = 0; l < layerCount; l++) {
                composed[l] = candidateWeights[l][order[l][tuple[l]]];
            }
            MatrixKey key = new MatrixKey(composed);
            Double cached = cache.get(key);
            if (cached != null) {
                double e = cached;
                if (valid(e)) {
                    updateBest(tuple, e, paramsOf(tuple));
                }
                return e;
            }
            double e;
            if (evaluations >= cap) {
                return Double.NaN; // no budget left: do not execute
            }
            evaluations++;
            e = evaluator.evaluate(composed);
            if (!valid(e)) {
                e = Double.POSITIVE_INFINITY;
            }
            cache.put(key, e);
            updateBest(tuple, e, paramsOf(tuple));
            return e;
        }

        private static boolean valid(double e) {
            return !Double.isNaN(e);
        }

        /** Keeps the lowest error; ties broken by layer-selection sequence. */
        private void updateBest(int[] tuple, double e, long params) {
            if (bestTuple == null || e < bestError
                    || (e == bestError && rawSequenceLess(tuple, bestTuple))) {
                bestTuple = tuple.clone();
                bestError = e;
                bestParams = params;
            }
        }

        private boolean rawSequenceLess(int[] a, int[] b) {
            for (int l = 0; l < layerCount; l++) {
                int ra = order[l][a[l]];
                int rb = order[l][b[l]];
                if (ra != rb) {
                    return ra < rb;
                }
            }
            return false;
        }

        /** Greedy coordinate refinement: replace layers with finer candidates. */
        private void hillClimb(int[] tuple) {
            boolean improved = true;
            while (improved && !capReached()) {
                improved = false;
                double currentError = errorOfTuple(tuple);
                int[] bestMove = null;
                double bestMoveError = currentError;
                int[] layers = shuffledLayers();
                outer:
                for (int li = 0; li < layers.length; li++) {
                    int l = layers[li];
                    int current = tuple[l];
                    for (int pos = current + 1; pos < widths[l]; pos++) {
                        int[] cand = tuple.clone();
                        cand[l] = pos;
                        if (!feasible(cand)) {
                            continue;
                        }
                        double e = evaluate(cand);
                        if (Double.isNaN(e)) {
                            break outer;
                        }
                        if (e < bestMoveError) {
                            bestMoveError = e;
                            bestMove = cand;
                        }
                    }
                }
                if (bestMove != null) {
                    tuple = bestMove;
                    improved = true;
                }
            }
        }

        private double errorOfTuple(int[] tuple) {
            double[][][] composed = new double[layerCount][][];
            for (int l = 0; l < layerCount; l++) {
                composed[l] = candidateWeights[l][order[l][tuple[l]]];
            }
            Double cached = cache.get(new MatrixKey(composed));
            return cached == null ? Double.NaN : cached;
        }

        private int[] shuffledLayers() {
            int[] ls = new int[layerCount];
            for (int i = 0; i < layerCount; i++) {
                ls[i] = i;
            }
            for (int i = layerCount - 1; i > 0; i--) {
                int j = random.nextInt(i + 1);
                int tmp = ls[i];
                ls[i] = ls[j];
                ls[j] = tmp;
            }
            return ls;
        }

        /** Biased feasible sample; quadratic bias keeps it near coarse picks. */
        private int[] sampleFeasible() {
            for (int attempt = 0; attempt < SAMPLE_ATTEMPTS; attempt++) {
                int[] t = new int[layerCount];
                long sum = 0L;
                boolean ok = true;
                for (int l = 0; l < layerCount; l++) {
                    int w = widths[l];
                    int pos;
                    if (w == 1) {
                        pos = 0;
                    } else {
                        double u = random.nextDouble();
                        pos = (int) (w * u * u);
                        if (pos >= w) {
                            pos = w - 1;
                        }
                    }
                    t[l] = pos;
                    sum += candidateParams[l][order[l][pos]];
                    if (sum > problem.paramBudget()) {
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    return t;
                }
            }
            return null;
        }

        /**
         * Counts feasible combinations with a simple DP, clamped at
         * Integer.MAX_VALUE so the cap decides on very large spaces.
         */
        private int countFeasible() {
            long budget = problem.paramBudget();
            Map<Long, Long> dp = new HashMap<Long, Long>();
            dp.put(0L, 1L);
            for (int l = 0; l < layerCount; l++) {
                Map<Long, Long> next = new HashMap<Long, Long>();
                for (Map.Entry<Long, Long> e : dp.entrySet()) {
                    long base = e.getKey();
                    long ways = e.getValue();
                    for (int pos = 0; pos < widths[l]; pos++) {
                        long ns = base + candidateParams[l][order[l][pos]];
                        if (ns <= budget) {
                            Long prev = next.get(ns);
                            long merged = (prev == null ? 0L : prev) + ways;
                            next.put(ns, Math.min(merged, (long) Integer.MAX_VALUE));
                        }
                    }
                }
                dp = next;
                if (dp.isEmpty()) {
                    return 0;
                }
            }
            long total = 0L;
            for (long ways : dp.values()) {
                total += ways;
                if (total >= Integer.MAX_VALUE) {
                    return Integer.MAX_VALUE;
                }
            }
            return (int) total;
        }

        /**
         * Depth-first enumeration of feasible combinations in coarse-first
         * lexicographic order, pruned by the minimum parameter sum still
         * needed by the remaining layers.
         *
         * @return true iff every feasible combination was evaluated.
         */
        private boolean enumerate() {
            long[] suffixMin = new long[layerCount + 1];
            for (int l = layerCount - 1; l >= 0; l--) {
                suffixMin[l] = suffixMin[l + 1] + minParams[l];
            }
            return dfs(0, new int[layerCount], 0L, suffixMin);
        }

        private boolean dfs(int layer, int[] tuple, long prefixParams, long[] suffixMin) {
            if (capReached()) {
                return false;
            }
            if (layer == layerCount) {
                double e = evaluate(tuple);
                return !Double.isNaN(e);
            }
            for (int pos = 0; pos < widths[layer]; pos++) {
                long p = prefixParams + candidateParams[layer][order[layer][pos]];
                if (p + suffixMin[layer + 1] > problem.paramBudget()) {
                    continue;
                }
                tuple[layer] = pos;
                if (!dfs(layer + 1, tuple, p, suffixMin)) {
                    return false;
                }
            }
            return true;
        }
    }

    /** Identity key over the composed matrices: duplicate candidates collide. */
    private static final class MatrixKey {
        private final double[][][] matrices;
        private final int hash;

        MatrixKey(double[][][] matrices) {
            int h = 1;
            for (double[][] m : matrices) {
                for (double[] row : m) {
                    h = 31 * h + java.util.Arrays.hashCode(row);
                }
                h = 31 * h + m.length;
            }
            this.matrices = matrices;
            this.hash = h;
        }

        @Override
        public int hashCode() {
            return hash;
        }

        @Override
        public boolean equals(Object obj) {
            if (this == obj) {
                return true;
            }
            if (!(obj instanceof MatrixKey)) {
                return false;
            }
            MatrixKey other = (MatrixKey) obj;
            if (matrices.length != other.matrices.length) {
                return false;
            }
            for (int l = 0; l < matrices.length; l++) {
                double[][] a = matrices[l];
                double[][] b = other.matrices[l];
                if (a.length != b.length) {
                    return false;
                }
                for (int r = 0; r < a.length; r++) {
                    if (!java.util.Arrays.equals(a[r], b[r])) {
                        return false;
                    }
                }
            }
            return true;
        }
    }
}
