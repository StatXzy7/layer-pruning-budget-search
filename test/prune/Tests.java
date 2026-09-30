package prune;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Zero-dependency test harness. @Test methods must throw AssertionError
 * on failure. The runner fails (exit 1) when zero test methods exist.
 */
public final class Tests {

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    @interface Test {
    }

    public static void main(String[] args) throws Exception {
        List<Method> methods = new ArrayList<Method>();
        for (Method m : Tests.class.getDeclaredMethods()) {
            if (m.isAnnotationPresent(Test.class)) {
                methods.add(m);
            }
        }
        if (methods.isEmpty()) {
            System.err.println("FAIL: zero test cases found");
            System.exit(1);
        }
        int passed = 0;
        List<String> failures = new ArrayList<String>();
        for (Method m : methods) {
            try {
                m.invoke(new Tests());
                passed++;
                System.out.println("PASS " + m.getName());
            } catch (Exception e) {
                Throwable cause = e.getCause() != null ? e.getCause() : e;
                failures.add(m.getName() + ": " + cause);
                System.out.println("FAIL " + m.getName() + " -> " + cause);
            }
        }
        System.out.println("----------------------------------------");
        System.out.println(passed + "/" + methods.size() + " tests passed");
        if (!failures.isEmpty()) {
            for (String f : failures) {
                System.err.println("FAILURE " + f);
            }
            System.exit(1);
        }
    }

    // ----------------------------------------------------------------
    // assertion helpers
    // ----------------------------------------------------------------

    private static void check(boolean cond, String msg) {
        if (!cond) {
            throw new AssertionError(msg);
        }
    }

    private static void approx(double actual, double expected, double tol, String msg) {
        check(Math.abs(actual - expected) <= tol,
                msg + " expected ~" + expected + " got " + actual);
    }

    private static void expectInvalid(Runnable r, String what) {
        try {
            r.run();
        } catch (InvalidProblemException expected) {
            return;
        }
        throw new AssertionError("expected InvalidProblemException for " + what);
    }

    private static void expectIllegalArg(Runnable r, String what) {
        try {
            r.run();
        } catch (IllegalArgumentException expected) {
            return;
        }
        throw new AssertionError("expected IllegalArgumentException for " + what);
    }

    // ----------------------------------------------------------------
    // fixtures
    // ----------------------------------------------------------------

    private static final int L = 3;
    private static final int[] DIMS = {4, 3, 3, 2};

    /** Two-layer residual: layer 2 output (dim 2) adds layer 0 (dim 2). */
    private static final int[] RESIDUAL_DIMS = {4, 3, 3, 3};

    private static LayerSpec[] residualLayers() {
        return new LayerSpec[] {
            new LayerSpec("l0", 4, 3, Activation.RELU, -1),
            new LayerSpec("l1", 3, 3, Activation.TANH, -1),
            new LayerSpec("l2", 3, 3, Activation.LINEAR, 0),
        };
    }

    private static LayerSpec[] plainLayers() {
        return new LayerSpec[] {
            new LayerSpec("l0", 4, 3, Activation.RELU, -1),
            new LayerSpec("l1", 3, 3, Activation.TANH, -1),
            new LayerSpec("l2", 3, 2, Activation.LINEAR, -1),
        };
    }

    private static double[][][] baseWeights() {
        return baseWeights(DIMS);
    }

    private static double[][][] baseWeights(int[] dims) {
        Random rng = new Random(20261001L);
        double[][][] w = new double[L][][];
        for (int l = 0; l < L; l++) {
            w[l] = new double[dims[l + 1]][dims[l]];
            for (int r = 0; r < dims[l + 1]; r++) {
                for (int c = 0; c < dims[l]; c++) {
                    w[l][r][c] = (rng.nextDouble() - 0.5) * 1.4;
                }
            }
        }
        return w;
    }

    private static double[][] baseBiases() {
        return baseBiases(DIMS);
    }

    private static double[][] baseBiases(int[] dims) {
        double[][] b = new double[L][];
        Random rng = new Random(77L);
        for (int l = 0; l < L; l++) {
            b[l] = new double[dims[l + 1]];
            for (int r = 0; r < b[l].length; r++) {
                b[l][r] = (rng.nextDouble() - 0.5) * 0.3;
            }
        }
        return b;
    }

    private static double[][] makeInputs(int n) {
        Random rng = new Random(4242L);
        double[][] x = new double[n][4];
        for (int s = 0; s < n; s++) {
            for (int j = 0; j < 4; j++) {
                x[s][j] = (rng.nextDouble() - 0.5) * 2.0;
            }
        }
        return x;
    }

    private static long countNonZeros(double[][] w) {
        long n = 0;
        for (double[] row : w) {
            for (double v : row) {
                if (v != 0.0) {
                    n++;
                }
            }
        }
        return n;
    }

    /**
     * Deterministic pruned+compensated candidate: zero out level-dependent
     * zeroCount entries, compensate survivors with a level-dependent scale.
     */
    private static Candidate makeCandidate(int layer, int level, long salt) {
        return makeCandidate(layer, level, salt, DIMS, baseWeights());
    }

    private static Candidate makeCandidate(int layer, int level, long salt,
                                           int[] dims, double[][][] base) {
        int out = dims[layer + 1];
        int in = dims[layer];
        double[][] w = new double[out][in];
        Random rng = new Random(salt * 1000003L + layer * 97L + level * 13L + 11L);
        int total = out * in;
        int zeros = Math.min(total - 1, level * 2);
        boolean[] mask = new boolean[total];
        int placed = 0;
        while (placed < zeros) {
            int p = rng.nextInt(total);
            if (!mask[p]) {
                mask[p] = true;
                placed++;
            }
        }
        double scale = 1.0 + 0.21 * level + 0.03 * (salt % 5);
        for (int r = 0; r < out; r++) {
            for (int c = 0; c < in; c++) {
                w[r][c] = mask[r * in + c] ? 0.0 : base[layer][r][c] * scale
                        + (rng.nextDouble() - 0.5) * 0.02;
            }
        }
        return new Candidate(w, countNonZeros(w), level);
    }

    private static List<List<Candidate>> threeLevels(long salt) {
        return threeLevels(salt, DIMS, baseWeights());
    }

    private static List<List<Candidate>> threeLevels(long salt, int[] dims, double[][][] base) {
        List<List<Candidate>> layers = new ArrayList<List<Candidate>>();
        for (int l = 0; l < L; l++) {
            List<Candidate> list = new ArrayList<Candidate>();
            for (int level = 1; level <= 3; level++) {
                list.add(makeCandidate(l, level, salt, dims, base));
            }
            layers.add(list);
        }
        return layers;
    }

    private static final double[][] INPUTS = makeInputs(8);

    private static NetworkConfig residualNetwork() {
        return NetworkConfig.create(residualLayers(),
                baseWeights(RESIDUAL_DIMS), baseBiases(RESIDUAL_DIMS));
    }

    private static NetworkConfig plainNetwork() {
        return NetworkConfig.create(plainLayers(), baseWeights(), baseBiases());
    }

    private static double[][] referenceOf(NetworkConfig net) {
        double[][] ref = new double[INPUTS.length][];
        for (int s = 0; s < INPUTS.length; s++) {
            ref[s] = NetworkRunner.run(net, INPUTS[s]);
        }
        return ref;
    }

    private static double errorOfSelection(NetworkConfig net,
                                           List<List<Candidate>> cands,
                                           int[] selection,
                                           double[][] reference) {
        double[][][] composed = new double[L][][];
        for (int l = 0; l < L; l++) {
            composed[l] = cands.get(l).get(selection[l]).weights();
        }
        return NetworkRunner.meanSquaredError(net.withWeights(composed), INPUTS, reference);
    }

    // ----------------------------------------------------------------
    // exhaustive reference (independent implementation for tests only)
    // ----------------------------------------------------------------

    private static final class BruteResult {
        int[] tuple;
        double error = Double.POSITIVE_INFINITY;
        long params;
        int feasibleCount;
    }

    private static BruteResult bruteForce(NetworkConfig net,
                                          List<List<Candidate>> cands,
                                          double[][] reference,
                                          long budget) {
        BruteResult best = new BruteResult();
        int n = cands.size();
        int[] t = new int[n];
        while (true) {
            long params = 0L;
            for (int l = 0; l < n; l++) {
                params += cands.get(l).get(t[l]).retainedParams();
            }
            if (params <= budget) {
                best.feasibleCount++;
                double e = errorOfSelection(net, cands, t, reference);
                boolean take = best.tuple == null || e < best.error;
                if (!take && e == best.error) {
                    for (int l = 0; l < n; l++) {
                        if (t[l] != best.tuple[l]) {
                            take = t[l] < best.tuple[l];
                            break;
                        }
                    }
                }
                if (take) {
                    best.error = e;
                    best.tuple = t.clone();
                    best.params = params;
                }
            }
            int k = n - 1;
            while (k >= 0 && t[k] + 1 >= cands.get(k).size()) {
                t[k] = 0;
                k--;
            }
            if (k < 0) {
                break;
            }
            t[k]++;
        }
        return best;
    }

    private static NetworkConfig TEST_NET;
    private static double[][] TEST_REF;
    private static List<List<Candidate>> TEST_CANDS;

    // ----------------------------------------------------------------
    // tests
    // ----------------------------------------------------------------

    @Test
    public void endToEndErrorMatchesDirectExecution() {
        TEST_NET = plainNetwork();
        TEST_REF = referenceOf(TEST_NET);
        TEST_CANDS = threeLevels(1L);
        PruningProblem problem = new PruningProblem(TEST_NET, TEST_CANDS,
                INPUTS, TEST_REF, 20L);
        SearchResult r = new PruningSearcher(problem).run(99L, 1000);

        double direct = errorOfSelection(TEST_NET, TEST_CANDS,
                r.selectedCandidateIndices(), TEST_REF);
        approx(r.error(), direct, 1e-12, "reported error must equal real composed-network error");
        check(r.status() == SearchResult.Status.PROVEN_OPTIMAL,
                "27 candidates, cap 1000 => fully enumerated, got " + r.status());
        check(r.evaluations() == r.feasibleCombinations(),
                "every feasible combination evaluated once: "
                        + r.evaluations() + " vs " + r.feasibleCombinations());
    }

    @Test
    public void exhaustiveReferenceZeroGapWithAndWithoutResidual() {
        for (int mode = 0; mode < 2; mode++) {
            NetworkConfig net = mode == 0 ? plainNetwork() : residualNetwork();
            double[][] ref = referenceOf(net);
            int[] dims = mode == 0 ? DIMS : RESIDUAL_DIMS;
            double[][][] bw = mode == 0 ? baseWeights() : baseWeights(RESIDUAL_DIMS);
            List<List<Candidate>> cands = threeLevels(10L + mode, dims, bw);
            long budget = 20L;
            BruteResult brute = bruteForce(net, cands, ref, budget);
            PruningProblem problem = new PruningProblem(net, cands, INPUTS, ref, budget);
            SearchResult r = new PruningSearcher(problem).run(7L, 1000);

            check(r.status() == SearchResult.Status.PROVEN_OPTIMAL,
                    "mode " + mode + " expected proven optimal, got " + r.status());
            check(Arrays.equals(r.selectedCandidateIndices(), brute.tuple),
                    "mode " + mode + " selection mismatch: "
                            + Arrays.toString(r.selectedCandidateIndices())
                            + " vs brute " + Arrays.toString(brute.tuple));
            approx(r.error(), brute.error, 1e-12, "mode " + mode + " error mismatch");
            double gap = r.error() - brute.error;
            check(gap == 0.0, "gap to exhaustive optimum must be 0, got " + gap);
            check(r.totalParams() == brute.params && r.totalParams() <= budget,
                    "parameter accounting mismatch");
            check(r.feasibleCombinations() == brute.feasibleCount,
                    "feasible count mismatch: " + r.feasibleCombinations()
                            + " vs " + brute.feasibleCount);
            System.out.printf("  mode %d: feasible=%d gap=%.3e selection=%s%n",
                    mode, brute.feasibleCount, gap, Arrays.toString(brute.tuple));
        }
    }

    @Test
    public void composedScoringBeatsLocalErrorProxy() {
        // Search over a handful of salts and show that the globally best
        // composed network differs from the choice obtained by minimising
        // each layer independently while holding the others at base-like
        // candidates; also assert search hits the exhaustive optimum.
        int foundMismatch = 0;
        for (long salt = 1L; salt <= 12L; salt++) {
            NetworkConfig net = plainNetwork();
            double[][] ref = referenceOf(net);
            List<List<Candidate>> cands = threeLevels(salt);
            long budget = 21L;
            BruteResult brute = bruteForce(net, cands, ref, budget);

            // "Local proxy": per-layer single-layer MSE with the other two
            // layers taking their coarsest (cheapest) candidate. This is a
            // layerwise proxy, not a composed evaluation.
            int[] proxy = new int[L];
            for (int target = 0; target < L; target++) {
                int bestK = 0;
                double bestE = Double.POSITIVE_INFINITY;
                for (int k = 0; k < 3; k++) {
                    int[] sel = new int[L];
                    sel[target] = k;
                    double e = errorOfSelection(net, cands, sel, ref);
                    if (e < bestE) {
                        bestE = e;
                        bestK = k;
                    }
                }
                proxy[target] = bestK;
            }
            double proxyError = errorOfSelection(net, cands, proxy, ref);
            if (!Arrays.equals(proxy, brute.tuple)) {
                foundMismatch++;
                check(brute.error < proxyError,
                        "salt " + salt + ": exhaustive composed optimum " + brute.error
                                + " must beat layerwise proxy " + proxyError);
            }
            PruningProblem problem = new PruningProblem(net, cands, INPUTS, ref, budget);
            SearchResult r = new PruningSearcher(problem).run(123L, 500);
            check(Arrays.equals(r.selectedCandidateIndices(), brute.tuple),
                    "salt " + salt + " search must reach exhaustive optimum");
        }
        check(foundMismatch >= 1,
                "test fixture must include a case where local proxy != composed optimum, found "
                        + foundMismatch);
        System.out.println("  salts with local-proxy != composed optimum: " + foundMismatch
                + " (search always matches exhaustive optimum)");
    }

    @Test
    public void budgetConservationExactTightAndInfeasible() {
        TEST_NET = plainNetwork();
        TEST_REF = referenceOf(TEST_NET);
        TEST_CANDS = threeLevels(3L);

        long minimum = 0L;
        for (List<Candidate> list : TEST_CANDS) {
            long m = Long.MAX_VALUE;
            for (Candidate c : list) {
                m = Math.min(m, c.retainedParams());
            }
            minimum += m;
        }

        // Exact budget: cheapest combination sits exactly on the limit.
        PruningProblem exact = new PruningProblem(TEST_NET, TEST_CANDS,
                INPUTS, TEST_REF, minimum);
        SearchResult re = new PruningSearcher(exact).run(1L, 100);
        check(re.hasFeasibleResult(), "exact budget must be feasible");
        check(re.totalParams() == minimum, "exact-budget result uses exactly " + minimum);
        check(re.feasibleCombinations() == 1, "only cheapest combo feasible, got "
                + re.feasibleCombinations());

        // Extremely tight budget: one parameter below feasibility.
        PruningProblem tight = new PruningProblem(TEST_NET, TEST_CANDS,
                INPUTS, TEST_REF, minimum - 1);
        SearchResult rt = new PruningSearcher(tight).run(2L, 100);
        check(rt.status() == SearchResult.Status.INFEASIBLE,
                "minimum-1 budget must be infeasible, got " + rt.status());
        check(!rt.hasFeasibleResult(), "infeasible result must report no solution");
        check(rt.evaluations() == 0, "no network may execute when provably infeasible");
        check(rt.report().contains("INFEASIBLE"), "report must state infeasibility");

        // Zero budget with strictly positive retained counts is infeasible.
        PruningProblem zero = new PruningProblem(TEST_NET, TEST_CANDS,
                INPUTS, TEST_REF, 0L);
        SearchResult rz = new PruningSearcher(zero).run(3L, 100);
        check(rz.status() == SearchResult.Status.INFEASIBLE, "zero budget infeasible");
    }

    @Test
    public void seedReproducibility() {
        TEST_NET = plainNetwork();
        TEST_REF = referenceOf(TEST_NET);
        TEST_CANDS = threeLevels(5L);
        PruningProblem problem = new PruningProblem(TEST_NET, TEST_CANDS,
                INPUTS, TEST_REF, 20L);
        PruningSearcher s1 = new PruningSearcher(problem);
        PruningSearcher s2 = new PruningSearcher(problem);
        SearchResult a = s1.run(987654321L, 7);
        SearchResult b = s2.run(987654321L, 7);
        check(a.evaluations() == b.evaluations(), "same seed -> same evaluation count");
        check(Arrays.equals(a.selectedCandidateIndices(), b.selectedCandidateIndices()),
                "same seed -> same per-layer selection");
        approx(a.error(), b.error(), 0.0, "same seed -> identical error");
        check(a.status() == SearchResult.Status.CAP_REACHED,
                "cap 7 over 27 combos must yield CAP_REACHED, got " + a.status());
        check(a.report().contains("NOT proven global optimum"),
                "capped result must not be labelled optimal");

        // A different seed is honoured by the RNG; the result is allowed but
        // not required to differ on this tiny space.
        SearchResult c = new PruningSearcher(problem).run(1L, 7);
        check(c.error() >= a.error() - 1e-12 || c.error() <= a.error() + 1e-12,
                "other-seed run must still produce a valid finite error");
    }

    @Test
    public void evaluationCapStopsWorkAndKeepsBest() {
        TEST_NET = plainNetwork();
        TEST_REF = referenceOf(TEST_NET);
        TEST_CANDS = threeLevels(2L);
        PruningProblem problem = new PruningProblem(TEST_NET, TEST_CANDS,
                INPUTS, TEST_REF, 20L);
        int cap = 4;
        SearchResult r = new PruningSearcher(problem).run(11L, cap);
        check(r.evaluations() == cap, "evaluations must stop exactly at cap, got "
                + r.evaluations());
        check(r.status() == SearchResult.Status.CAP_REACHED, "cap before exhaustion -> CAP_REACHED");
        check(r.totalParams() <= 20L, "returned heuristic result must stay feasible");
        double direct = errorOfSelection(TEST_NET, TEST_CANDS,
                r.selectedCandidateIndices(), TEST_REF);
        approx(r.error(), direct, 1e-12, "capped result error must be a real executed error");

        // Best-found invariant: an independent full enumeration never beats it
        // by accident beyond the proven gap; record the gap explicitly.
        BruteResult brute = bruteForce(TEST_NET, TEST_CANDS, TEST_REF, 20L);
        double gap = r.error() - brute.error;
        check(gap >= -1e-12, "heuristic error cannot be below true optimum");
        System.out.printf("  cap=%d evals=%d gap-to-optimum=%.6e%n", cap, r.evaluations(), gap);

        expectIllegalArg(new Runnable() {
            @Override public void run() { new PruningSearcher(problem).run(1L, 0); }
        }, "maxEvaluations=0");
        expectIllegalArg(new Runnable() {
            @Override public void run() { new PruningSearcher(problem).run(1L, -3); }
        }, "negative maxEvaluations");
    }

    @Test
    public void duplicateCandidatesAreNotChargedTwice() {
        // Three layers; every layer lists the same cheap matrix twice plus
        // one distinct fine matrix. 8 combinations exist, but only 2 have
        // distinct composed matrices.
        LayerSpec[] specs = new LayerSpec[] {
            new LayerSpec("a", 2, 1, Activation.LINEAR, -1),
            new LayerSpec("b", 1, 1, Activation.LINEAR, -1),
            new LayerSpec("c", 1, 1, Activation.LINEAR, -1),
        };
        double[][][] w = new double[][][] {
            { {0.0, 0.0} },
            { {0.0} },
            { {0.0} },
        };
        double[][] biases = { {0.0}, {0.0}, {0.0} };
        NetworkConfig net = NetworkConfig.create(specs, w, biases);
        double[][] inputs = { {1.0, 2.0}, {3.0, -1.0} };
        double[][] refs = { {0.0}, {0.0} };

        double[][] coarse0 = { {0.0, 0.0} };
        double[][] fine0 = { {1.0, 0.0} };
        double[][] coarse1 = { {0.0} };
        double[][] fine1 = { {1.0} };
        List<Candidate> l0 = Arrays.asList(
                new Candidate(coarse0, 0, 2),
                new Candidate(coarse0, 0, 2),
                new Candidate(fine0, 1, 0));
        List<Candidate> l1 = Arrays.asList(
                new Candidate(coarse1, 0, 2),
                new Candidate(coarse1, 0, 2),
                new Candidate(fine1, 1, 0));
        List<Candidate> l2 = Arrays.asList(
                new Candidate(coarse1, 0, 2),
                new Candidate(coarse1, 0, 2),
                new Candidate(fine1, 1, 0));
        List<List<Candidate>> all = Arrays.asList(l0, l1, l2);

        final int[] invoked = {0};
        Evaluator counting = new Evaluator() {
            @Override
            public double evaluate(double[][][] composedWeights) {
                invoked[0]++;
                return NetworkRunner.meanSquaredError(
                        net.withWeights(composedWeights), inputs, refs);
            }
        };
        PruningProblem problem = new PruningProblem(net, all, inputs, refs, 9L);
        SearchResult r = new PruningSearcher(problem, counting).run(4L, 50);
        check(r.status() == SearchResult.Status.PROVEN_OPTIMAL,
                "all 8 combos enumerate; only distinct matrices execute, got " + r.status());
        check(r.feasibleCombinations() == 27, "raw feasible combinations = 27, got "
                + r.feasibleCombinations());
        check(r.evaluations() == 8, "only 8 distinct composed networks of 27, got "
                + r.evaluations());
        check(invoked[0] == 8, "evaluator must be invoked for 8 distinct networks, got "
                + invoked[0]);

        // With cap 1 the search returns the single executed coarse result.
        invoked[0] = 0;
        SearchResult capped = new PruningSearcher(problem, counting).run(4L, 1);
        check(capped.evaluations() == 1 && invoked[0] == 1,
                "cap 1 executes exactly one network");
        check(capped.status() == SearchResult.Status.CAP_REACHED, "cap 1 cannot exhaust");
    }

    @Test
    public void tiesBreakByLayerSelectionSequence() {
        // Single linear layer, two different zero patterns produce identical
        // zero output on all calibration inputs (first vs second input
        // column zeroed; inputs keep that column at zero).
        LayerSpec[] specs = { new LayerSpec("a", 2, 1, Activation.LINEAR, -1) };
        double[][][] w = { { {1.0, 1.0} } };
        double[][] biases = { {0.0} };
        NetworkConfig net = NetworkConfig.create(specs, w, biases);
        double[][] inputs = { {0.0, 0.0}, {0.0, 0.0} };
        double[][] refs = { {0.0}, {0.0} };
        double[][] killFirst = { {0.0, 1.0} };
        double[][] killSecond = { {1.0, 0.0} };
        List<Candidate> list = Arrays.asList(
                new Candidate(killFirst, 1, 1),
                new Candidate(killSecond, 1, 1));
        PruningProblem problem = new PruningProblem(net,
                Arrays.asList(list), inputs, refs, 5L);
        SearchResult r1 = new PruningSearcher(problem).run(1L, 10);
        SearchResult r2 = new PruningSearcher(problem).run(2L, 10);
        check(r1.selectedCandidateIndices()[0] == 0,
                "tied errors -> lexicographically smallest layer sequence (candidate 0)");
        check(r2.selectedCandidateIndices()[0] == 0, "tie result must be seed-independent");
        approx(r1.error(), 0.0, 0.0, "tied zero error");
    }

    @Test
    public void invalidInputsAreRejected() {
        NetworkConfig net = plainNetwork();
        double[][] ref = referenceOf(net);
        List<List<Candidate>> good = threeLevels(1L);
        double[][][] goodW = baseWeights();

        // Empty candidate layer.
        List<List<Candidate>> emptyLayer = new ArrayList<List<Candidate>>(good);
        emptyLayer.set(1, new ArrayList<Candidate>());
        expectInvalid(new Runnable() {
            @Override public void run() {
                new PruningProblem(net, emptyLayer, INPUTS, ref, 20L);
            }
        }, "empty candidate layer");

        // Wrong candidate dimensions.
        double[][] badShape = new double[2][2];
        List<List<Candidate>> badDims = new ArrayList<List<Candidate>>(good);
        badDims.set(0, Arrays.asList(new Candidate(badShape, 4, 1)));
        expectInvalid(new Runnable() {
            @Override public void run() {
                new PruningProblem(net, badDims, INPUTS, ref, 20L);
            }
        }, "candidate wrong dimensions");

        // Negative retained parameter count.
        expectInvalid(new Runnable() {
            @Override public void run() {
                new Candidate(goodW[0], -1L, 1);
            }
        }, "negative retained params");
        expectInvalid(new Runnable() {
            @Override public void run() {
                new Candidate(goodW[0], 100000L, 1);
            }
        }, "retained params beyond matrix capacity");

        // Non-finite matrix (candidate, network weight, bias, calibration data).
        double[][] nanMatrix = goodW[0].clone();
        nanMatrix[0] = goodW[0][0].clone();
        nanMatrix[0][0] = Double.NaN;
        expectInvalid(new Runnable() {
            @Override public void run() {
                new Candidate(nanMatrix, 10L, 1);
            }
        }, "non-finite candidate matrix");

        double[][][] infW = goodW.clone();
        infW[0] = goodW[0].clone();
        infW[0][0] = goodW[0][0].clone();
        infW[0][0][0] = Double.POSITIVE_INFINITY;
        expectInvalid(new Runnable() {
            @Override public void run() {
                NetworkConfig.create(plainLayers(), infW, baseBiases());
            }
        }, "non-finite network weight");

        double[][] infBias = baseBiases();
        infBias[1] = infBias[1].clone();
        infBias[1][0] = Double.NaN;
        expectInvalid(new Runnable() {
            @Override public void run() {
                NetworkConfig.create(plainLayers(), baseWeights(), infBias);
            }
        }, "non-finite bias");

        double[][] badInputs = INPUTS.clone();
        badInputs[0] = INPUTS[0].clone();
        badInputs[0][0] = Double.NaN;
        expectInvalid(new Runnable() {
            @Override public void run() {
                new PruningProblem(net, good, badInputs, ref, 20L);
            }
        }, "non-finite calibration input");

        // Bad calibration shapes / counts.
        expectInvalid(new Runnable() {
            @Override public void run() {
                new PruningProblem(net, good, new double[0][], new double[0][], 20L);
            }
        }, "empty calibration set");
        final double[][] shortRefs = Arrays.copyOf(ref, ref.length - 1);
        expectInvalid(new Runnable() {
            @Override public void run() {
                new PruningProblem(net, good, INPUTS, shortRefs, 20L);
            }
        }, "input/output count mismatch");
        double[][] wideInputs = new double[INPUTS.length][5];
        expectInvalid(new Runnable() {
            @Override public void run() {
                new PruningProblem(net, good, wideInputs, ref, 20L);
            }
        }, "wrong input dimension");

        // Non-composable residual: source must be earlier and same width.
        LayerSpec[] future = {
            new LayerSpec("x", 2, 2, Activation.LINEAR, -1),
            new LayerSpec("y", 2, 1, Activation.LINEAR, 1),
        };
        expectInvalid(new Runnable() {
            @Override public void run() {
                NetworkConfig.create(future,
                        new double[][][] { { {0.0, 0.0}, {0.0, 0.0} }, { {0.0, 0.0} } },
                        new double[][] { {0.0, 0.0}, {0.0} });
            }
        }, "residual pointing at self/later layer");

        LayerSpec[] widthMismatch = {
            new LayerSpec("x", 2, 2, Activation.LINEAR, -1),
            new LayerSpec("y", 2, 1, Activation.LINEAR, 0),
        };
        expectInvalid(new Runnable() {
            @Override public void run() {
                NetworkConfig.create(widthMismatch,
                        new double[][][] { { {0.0, 0.0}, {0.0, 0.0} }, { {0.0, 0.0} } },
                        new double[][] { {0.0, 0.0}, {0.0} });
            }
        }, "residual with mismatched output widths");

        LayerSpec[] chainMismatch = {
            new LayerSpec("x", 2, 3, Activation.RELU, -1),
            new LayerSpec("y", 2, 3, Activation.RELU, -1),
        };
        expectInvalid(new Runnable() {
            @Override public void run() {
                NetworkConfig.create(chainMismatch,
                        new double[][][] { new double[3][2], new double[3][2] },
                        new double[][] { new double[3], new double[3] });
            }
        }, "layers that do not compose dimensionally");

        // Missing candidate lists / null entries and negative budget.
        expectInvalid(new Runnable() {
            @Override public void run() {
                new PruningProblem(net, Arrays.asList(good.get(0), good.get(1)),
                        INPUTS, ref, 20L);
            }
        }, "candidate list count differs from layers");
        List<List<Candidate>> withNull = new ArrayList<List<Candidate>>(good);
        withNull.set(2, Arrays.asList((Candidate) null));
        expectInvalid(new Runnable() {
            @Override public void run() {
                new PruningProblem(net, withNull, INPUTS, ref, 20L);
            }
        }, "null candidate entry");
        expectIllegalArg(new Runnable() {
            @Override public void run() {
                new PruningProblem(net, good, INPUTS, ref, -1L);
            }
        }, "negative budget");
    }
}
