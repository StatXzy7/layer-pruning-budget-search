package pruner;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Self-contained test suite (Java standard library only). Every method
 * annotated with {@link Test} is run reflectively; zero discovered tests is a
 * hard failure. Exhaustive brute-force references are used on the small
 * fixtures so the heuristic search's gap to the true optimum is measured.
 */
public final class PrunerTests {

    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    @interface Test {
    }

    private static int passed;
    private static int failed;
    private static final List<String> failureMessages = new ArrayList<>();

    public static void main(String[] args) throws Exception {
        List<Method> tests = new ArrayList<>();
        for (Method method : PrunerTests.class.getDeclaredMethods()) {
            if (method.isAnnotationPresent(Test.class)) {
                tests.add(method);
            }
        }
        if (tests.isEmpty()) {
            System.out.println("FAIL: no test cases discovered");
            System.exit(1);
        }
        Collections.sort(tests, new java.util.Comparator<Method>() {
            @Override
            public int compare(Method a, Method b) {
                return a.getName().compareTo(b.getName());
            }
        });

        PrunerTests instance = new PrunerTests();
        for (Method test : tests) {
            try {
                test.invoke(instance);
                passed++;
                System.out.println("PASS " + test.getName());
            } catch (InvocationTargetException exception) {
                failed++;
                Throwable cause = exception.getCause();
                String message = cause == null ? "unknown failure"
                        : cause.getClass().getSimpleName() + ": "
                                + cause.getMessage();
                failureMessages.add(test.getName() + " -> " + message);
                System.out.println("FAIL " + test.getName() + " -> "
                        + message);
            }
        }

        System.out.println();
        System.out.println("tests: " + (passed + failed)
                + ", passed: " + passed + ", failed: " + failed);
        if (failed > 0) {
            for (String message : failureMessages) {
                System.out.println("  " + message);
            }
            System.exit(1);
        }
    }

    // ------------------------------------------------------------------
    // Assertions
    // ------------------------------------------------------------------

    private static void check(boolean condition, String message) {
        if (!condition) {
            throw new AssertionError(message);
        }
    }

    private static void checkEquals(long expected, long actual,
                                    String message) {
        if (expected != actual) {
            throw new AssertionError(message + " expected <" + expected
                    + "> but was <" + actual + ">");
        }
    }

    private static void checkEquals(int expected, int actual,
                                    String message) {
        if (expected != actual) {
            throw new AssertionError(message + " expected <" + expected
                    + "> but was <" + actual + ">");
        }
    }

    private static void checkClose(double expected, double actual,
                                   double tolerance, String message) {
        if (!(Math.abs(expected - actual) <= tolerance)) {
            throw new AssertionError(message + " expected <" + expected
                    + "> but was <" + actual + "> (tolerance " + tolerance
                    + ")");
        }
    }

    private static void checkThrows(Class<? extends Throwable> expected,
                                    ThrowingRunnable runnable,
                                    String message) {
        try {
            runnable.run();
        } catch (Throwable thrown) {
            if (expected.isInstance(thrown)) {
                return;
            }
            throw new AssertionError(message + ": expected "
                    + expected.getSimpleName() + " but got "
                    + thrown.getClass().getSimpleName() + ": "
                    + thrown.getMessage());
        }
        throw new AssertionError(message + ": expected "
                + expected.getSimpleName() + " but nothing was thrown");
    }

    private interface ThrowingRunnable {
        void run() throws Exception;
    }

    // ------------------------------------------------------------------
    // Fixture helpers
    // ------------------------------------------------------------------

    private static LayerCandidate candidate(double[][] weights,
                                            double[] biases, int level) {
        return new LayerCandidate(weights, biases,
                (long) weights.length * weights[0].length + biases.length,
                level);
    }

    private static double[][] matrix(int outDim, int inDim,
                                     Random random) {
        double[][] matrix = new double[outDim][inDim];
        for (int out = 0; out < outDim; out++) {
            for (int in = 0; in < inDim; in++) {
                matrix[out][in] = 0.5 * random.nextGaussian();
            }
        }
        return matrix;
    }

    private static double[] vector(int dim, Random random) {
        double[] vector = new double[dim];
        for (int unit = 0; unit < dim; unit++) {
            vector[unit] = 0.2 * random.nextGaussian();
        }
        return vector;
    }

    // ------------------------------------------------------------------
    // Fixtures
    // ------------------------------------------------------------------

    private NetworkDefinition manualNet;
    private CalibrationData manualData;

    private NetworkDefinition chainNet;
    private CalibrationData chainData;

    private NetworkDefinition residualNet;
    private CalibrationData residualData;

    private NetworkDefinition seedNet;
    private CalibrationData seedData;

    private NetworkDefinition dedupNet;
    private CalibrationData dedupData;

    private NetworkDefinition tieNetA;
    private NetworkDefinition tieNetB;
    private CalibrationData tieData;

    private NetworkDefinition jointNet;
    private CalibrationData jointData;

    private void buildFixtures() {
        buildManualFixture();
        buildChainFixture();
        buildResidualFixture();
        buildSeedFixture();
        buildDedupFixture();
        buildTieFixtures();
        buildJointFixture();
    }

    private void buildManualFixture() {
        LayerCandidate pruned = candidate(
                new double[][]{{1.0, 1.0}, {2.0, 2.0}},
                new double[]{0.0, 0.0}, 1);
        LayerCandidate dense = candidate(
                new double[][]{{1.0, 2.0}, {3.0, 4.0}},
                new double[]{0.1, -0.2}, 0);
        manualNet = new NetworkDefinition(2,
                Collections.singletonList(
                        new LayerSpec(Activation.IDENTITY, null,
                                Arrays.asList(pruned, dense))));
        manualData = new CalibrationData(
                Arrays.asList(new double[]{1.0, 1.0},
                        new double[]{-2.0, 0.5}),
                Arrays.asList(new double[]{3.1, 6.8},
                        new double[]{-0.9, -4.2}));
    }

    private void buildChainFixture() {
        Random random = new Random(42);
        LayerCandidate l0p = candidate(matrix(1, 2, random),
                vector(1, random), 1);
        LayerCandidate l0d = candidate(matrix(2, 2, random),
                vector(2, random), 0);
        LayerCandidate l1p = candidate(matrix(2, 1, random),
                vector(2, random), 1);
        LayerCandidate l1d = candidate(matrix(2, 2, random),
                vector(2, random), 0);
        chainNet = new NetworkDefinition(2, Arrays.asList(
                new LayerSpec(Activation.RELU, null,
                        Arrays.asList(l0p, l0d)),
                new LayerSpec(Activation.IDENTITY, null,
                        Arrays.asList(l1p, l1d))));

        List<double[]> inputs = new ArrayList<>();
        Random sampleRandom = new Random(7);
        for (int sample = 0; sample < 4; sample++) {
            inputs.add(new double[]{sampleRandom.nextGaussian(),
                    sampleRandom.nextGaussian()});
        }
        ComposedNetwork dense = new ComposedNetwork(chainNet,
                new int[]{1, 1});
        List<double[]> refs = new ArrayList<>();
        for (double[] input : inputs) {
            refs.add(dense.predict(input));
        }
        chainData = new CalibrationData(inputs, refs);
    }

    private void buildResidualFixture() {
        Random random = new Random(99);
        LayerCandidate l0p = candidate(matrix(2, 3, random),
                vector(2, random), 1);
        LayerCandidate l0d = candidate(matrix(3, 3, random),
                vector(3, random), 0);
        LayerCandidate l1p = candidate(matrix(3, 2, random),
                vector(3, random), 1);
        LayerCandidate l1d = candidate(matrix(3, 3, random),
                vector(3, random), 0);
        LayerCandidate l2p = candidate(matrix(2, 3, random),
                vector(2, random), 1);
        LayerCandidate l2d = candidate(matrix(2, 3, random),
                vector(2, random), 0);
        residualNet = new NetworkDefinition(3, Arrays.asList(
                new LayerSpec(Activation.RELU, null,
                        Arrays.asList(l0p, l0d)),
                new LayerSpec(Activation.IDENTITY, -1,
                        Arrays.asList(l1p, l1d)),
                new LayerSpec(Activation.IDENTITY, null,
                        Arrays.asList(l2p, l2d))));

        List<double[]> inputs = new ArrayList<>();
        Random sampleRandom = new Random(11);
        for (int sample = 0; sample < 4; sample++) {
            inputs.add(vector(3, sampleRandom));
        }
        ComposedNetwork dense = new ComposedNetwork(residualNet,
                new int[]{1, 1, 1});
        List<double[]> refs = new ArrayList<>();
        for (double[] input : inputs) {
            refs.add(dense.predict(input));
        }
        residualData = new CalibrationData(inputs, refs);
    }

    private void buildSeedFixture() {
        Random random = new Random(123);
        List<LayerCandidate> layer0 = new ArrayList<>();
        List<LayerCandidate> layer1 = new ArrayList<>();
        for (int candidate = 0; candidate < 4; candidate++) {
            int level = candidate < 2 ? 1 : 0;
            layer0.add(candidate(matrix(2, 2, random),
                    vector(2, random), level));
            layer1.add(candidate(matrix(2, 2, random),
                    vector(2, random), level));
        }
        seedNet = new NetworkDefinition(2, Arrays.asList(
                new LayerSpec(Activation.RELU, null, layer0),
                new LayerSpec(Activation.IDENTITY, null, layer1)));

        List<double[]> inputs = new ArrayList<>();
        List<double[]> refs = new ArrayList<>();
        Random sampleRandom = new Random(9);
        Random refRandom = new Random(13);
        for (int sample = 0; sample < 4; sample++) {
            inputs.add(new double[]{sampleRandom.nextGaussian(),
                    sampleRandom.nextGaussian()});
            refs.add(new double[]{refRandom.nextGaussian(),
                    refRandom.nextGaussian()});
        }
        seedData = new CalibrationData(inputs, refs);
    }

    private void buildDedupFixture() {
        Random random = new Random(77);
        double[][] tensorA = matrix(2, 2, random);
        double[] biasA = vector(2, random);
        double[][] tensorB = matrix(2, 2, random);
        double[] biasB = vector(2, random);
        double[][] tensorC = matrix(2, 2, random);
        double[] biasC = vector(2, random);
        dedupNet = new NetworkDefinition(2,
                Collections.singletonList(
                        new LayerSpec(Activation.IDENTITY, null,
                                Arrays.asList(
                                        candidate(tensorA, biasA, 1),
                                        candidate(tensorB, biasB, 1),
                                        candidate(tensorC, biasC, 0),
                                        candidate(tensorA, biasA, 0)))));

        List<double[]> inputs = new ArrayList<>();
        List<double[]> refs = new ArrayList<>();
        Random sampleRandom = new Random(78);
        for (int sample = 0; sample < 4; sample++) {
            inputs.add(new double[]{sampleRandom.nextGaussian(),
                    sampleRandom.nextGaussian()});
            refs.add(new double[]{sampleRandom.nextGaussian(),
                    sampleRandom.nextGaussian()});
        }
        dedupData = new CalibrationData(inputs, refs);
    }

    private void buildTieFixtures() {
        LayerCandidate scale = candidate(
                new double[][]{{2.0, 0.0}, {0.0, 2.0}},
                new double[]{0.0, 0.0}, 0);
        LayerCandidate zero = candidate(
                new double[][]{{0.0, 0.0}, {0.0, 0.0}},
                new double[]{0.0, 0.0}, 0);
        tieNetA = new NetworkDefinition(2,
                Collections.singletonList(
                        new LayerSpec(Activation.IDENTITY, null,
                                Arrays.asList(scale, zero))));
        tieNetB = new NetworkDefinition(2,
                Collections.singletonList(
                        new LayerSpec(Activation.IDENTITY, null,
                                Arrays.asList(zero, scale))));
        tieData = new CalibrationData(
                Arrays.asList(new double[]{1.0, 1.0},
                        new double[]{-1.0, -1.0}),
                Arrays.asList(new double[]{1.0, 1.0},
                        new double[]{-1.0, -1.0}));
    }

    private void buildJointFixture() {
        LayerCandidate l0A = candidate(
                new double[][]{{1.0, 1.0}, {0.0, 1.0}},
                new double[]{0.0, 0.0}, 1);
        LayerCandidate l0B = candidate(
                new double[][]{{1.0, 0.0}, {0.0, 1.0}},
                new double[]{0.0, 0.0}, 0);
        LayerCandidate l1A = candidate(
                new double[][]{{1.0, 0.0}, {0.0, 1.0}},
                new double[]{0.0, 0.0}, 1);
        LayerCandidate l1B = candidate(
                new double[][]{{1.0, -1.0}, {0.0, 1.0}},
                new double[]{0.0, 0.0}, 0);
        jointNet = new NetworkDefinition(2, Arrays.asList(
                new LayerSpec(Activation.IDENTITY, null,
                        Arrays.asList(l0A, l0B)),
                new LayerSpec(Activation.IDENTITY, null,
                        Arrays.asList(l1A, l1B))));
        jointData = new CalibrationData(
                Arrays.asList(new double[]{1.0, 0.0},
                        new double[]{0.0, 1.0}),
                Arrays.asList(new double[]{1.0, 0.0},
                        new double[]{0.0, 1.0}));
    }

    // ------------------------------------------------------------------
    // Tests: end-to-end scoring (genuine composed execution)
    // ------------------------------------------------------------------

    @Test
    public void manualSingleLayerScoresTrueEndToEndError() {
        buildFixtures();
        ComposedNetwork pruned = new ComposedNetwork(manualNet,
                new int[]{0});
        ComposedNetwork dense = new ComposedNetwork(manualNet,
                new int[]{1});
        Scorer scorer = new Scorer(manualData);
        checkClose(2.7125, scorer.meanSquaredError(pruned), 1e-12,
                "pruned single-layer end-to-end MSE");
        checkClose(0.0, scorer.meanSquaredError(dense), 1e-12,
                "dense single-layer reproduces references");

        SearchResult result = new PruningSearcher(manualNet, manualData)
                .search(100, 1, 1000);
        check(result.isFeasible(), "unrestricted search must be feasible");
        checkClose(0.0, result.error(), 1e-12,
                "search selects zero-error candidate");
        check(Arrays.equals(new int[]{1}, result.selection()),
                "selection must be the dense candidate");
    }

    @Test
    public void residualAdditionExecutesPerConfiguration() {
        buildFixtures();
        ComposedNetwork dense = new ComposedNetwork(residualNet,
                new int[]{1, 1, 1});
        double[] input = residualData.inputs().get(0);
        double[] manual = independentResidualForward(input);
        double[] actual = dense.predict(input);
        check(actual.length == manual.length,
                "residual output length");
        for (int unit = 0; unit < manual.length; unit++) {
            checkClose(manual[unit], actual[unit], 1e-10,
                    "independent residual forward at unit " + unit);
        }
        checkClose(0.0,
                new Scorer(residualData).meanSquaredError(dense),
                1e-10, "dense residual network reproduces references");
    }

    private double[] independentResidualForward(double[] input) {
        LayerCandidate l0 = residualNet.layers().get(0).candidates().get(1);
        LayerCandidate l1 = residualNet.layers().get(1).candidates().get(1);
        LayerCandidate l2 = residualNet.layers().get(2).candidates().get(1);

        double[] h0 = affine(l0, input);
        for (int unit = 0; unit < h0.length; unit++) {
            h0[unit] = Activation.RELU.apply(h0[unit]);
        }
        double[] h1 = affine(l1, h0);
        for (int unit = 0; unit < h1.length; unit++) {
            h1[unit] = h1[unit] + input[unit];
        }
        double[] h2 = affine(l2, h1);
        return h2;
    }

    private static double[] affine(LayerCandidate candidate, double[] x) {
        double[][] weights = candidate.weights();
        double[] biases = candidate.biases();
        double[] out = new double[candidate.outDim()];
        for (int i = 0; i < out.length; i++) {
            double sum = biases[i];
            for (int j = 0; j < x.length; j++) {
                sum += weights[i][j] * x[j];
            }
            out[i] = sum;
        }
        return out;
    }

    @Test
    public void jointScoringBeatsSumOfLocalErrors() {
        buildFixtures();
        ComposedNetwork ab = new ComposedNetwork(jointNet,
                new int[]{0, 1});
        Scorer scorer = new Scorer(jointData);

        double firstOnly = standaloneMeanSquaredError(jointNet, jointData,
                0, 0);
        double secondOnly = standaloneMeanSquaredError(jointNet, jointData,
                1, 1);
        check(firstOnly > 0.0 && secondOnly > 0.0,
                "each layer locally is imperfect, so a sum-of-local-error "
                        + "heuristic would not identify the perfect pairing");
        checkClose(0.0, scorer.meanSquaredError(ab), 1e-12,
                "composed cross-layer pairing reproduces identity "
                        + "end-to-end");

        SearchResult result = new PruningSearcher(jointNet, jointData)
                .search(100, 1, 1000);
        checkClose(0.0, result.error(), 1e-12,
                "search must find the zero-error composed pairing");

        ExhaustiveReference.Optimum optimum =
                ExhaustiveReference.optimize(jointNet, jointData, 100);
        checkClose(optimum.error(), result.error(), 1e-12,
                "search matches exhaustive optimum");
        System.out.println("  [joint] search error=" + result.error()
                + ", exhaustive reference=" + optimum.error()
                + ", gap=" + (result.error() - optimum.error()));
    }

    private double standaloneMeanSquaredError(NetworkDefinition network,
                                             CalibrationData data,
                                             int layer, int candidate) {
        LayerCandidate chosen =
                network.layers().get(layer).candidates().get(candidate);
        double sum = 0.0;
        long count = 0;
        for (int sample = 0; sample < data.size(); sample++) {
            double[] x = data.inputs().get(sample);
            double[] y = data.referenceOutputs().get(sample);
            double[] predicted;
            if (layer == 0) {
                predicted = affine(chosen, x);
            } else {
                predicted = affine(chosen, y);
            }
            for (int unit = 0; unit < y.length; unit++) {
                double difference = predicted[unit] - y[unit];
                sum += difference * difference;
            }
            count += y.length;
        }
        return sum / count;
    }

    // ------------------------------------------------------------------
    // Tests: exhaustive reference comparison and budget conservation
    // ------------------------------------------------------------------

    @Test
    public void chainNetworkExhaustiveReferenceGap() {
        buildFixtures();
        long denseBudget = 12L;
        long prunedBudget = 8L;
        long impossibleBudget = 6L;

        ExhaustiveReference.Optimum denseOptimum =
                ExhaustiveReference.optimize(chainNet, chainData,
                        denseBudget);
        ExhaustiveReference.Optimum prunedOptimum =
                ExhaustiveReference.optimize(chainNet, chainData,
                        prunedBudget);
        ExhaustiveReference.Optimum emptyOptimum =
                ExhaustiveReference.optimize(chainNet, chainData,
                        impossibleBudget);

        check(denseOptimum.isFeasible(), "dense exhaustive is feasible");
        check(prunedOptimum.isFeasible(), "pruned exhaustive is feasible");
        check(!emptyOptimum.isFeasible(),
                "below minimum params exhaustive is infeasible");
        checkEquals(2, denseOptimum.feasibleEvaluations(),
                "two distinct feasible tensor schemes at dense budget");
        checkEquals(1, prunedOptimum.feasibleEvaluations(),
                "only pruned scheme fits the tight budget");

        PruningSearcher searcher = new PruningSearcher(chainNet, chainData);
        SearchResult denseResult = searcher.search(denseBudget, 5, 1000);
        SearchResult prunedResult = searcher.search(prunedBudget, 5, 1000);
        SearchResult emptyResult = searcher.search(impossibleBudget, 5,
                1000);

        checkClose(denseOptimum.error(), denseResult.error(), 1e-12,
                "dense search equals exhaustive reference");
        check(Arrays.equals(denseOptimum.selection(),
                denseResult.selection()),
                "dense search selection equals exhaustive optimum");
        checkClose(prunedOptimum.error(), prunedResult.error(), 1e-12,
                "pruned search equals exhaustive reference");
        check(Arrays.equals(prunedOptimum.selection(),
                prunedResult.selection()),
                "pruned search selection equals exhaustive optimum");
        check(!emptyResult.isFeasible(),
                "search reports no feasible scheme under tight budget");

        System.out.println("  [chain dense] search=" + denseResult.error()
                + ", exhaustive=" + denseOptimum.error()
                + ", gap=" + (denseResult.error() - denseOptimum.error()));
        System.out.println("  [chain tight] search=" + prunedResult.error()
                + ", exhaustive=" + prunedOptimum.error()
                + ", gap=" + (prunedResult.error()
                        - prunedOptimum.error()));
    }

    @Test
    public void residualNetworkExhaustiveReferenceGap() {
        buildFixtures();
        long budget = 32L;
        ExhaustiveReference.Optimum optimum =
                ExhaustiveReference.optimize(residualNet, residualData,
                        budget);
        check(optimum.isFeasible(), "residual exhaustive is feasible");
        checkEquals(4, optimum.composableSchemes(),
                "four raw residual selections are dimensionally composable");
        checkEquals(4, optimum.feasibleEvaluations(),
                "four distinct feasible residual tensor schemes fit");
        ExhaustiveReference.Optimum tightResidual =
                ExhaustiveReference.optimize(residualNet, residualData,
                        25L);
        checkEquals(2, tightResidual.feasibleEvaluations(),
                "two distinct schemes fit the tighter residual budget");
        check(!ExhaustiveReference.optimize(residualNet, residualData,
                        24L).isFeasible(),
                "one below the cheapest residual scheme is infeasible");

        SearchResult uncapped =
                new PruningSearcher(residualNet, residualData)
                        .search(budget, 3, 1000);
        checkClose(optimum.error(), uncapped.error(), 1e-10,
                "uncapped search matches exhaustive residual optimum");
        check(Arrays.equals(optimum.selection(), uncapped.selection()),
                "uncapped residual selection equals optimum");

        SearchResult capped =
                new PruningSearcher(residualNet, residualData)
                        .search(budget, 3, 1);
        check(capped.isFeasible(), "anchor evaluation is feasible");
        checkEquals(1, capped.evaluations(),
                "capped residual search evaluates exactly once");
        check(capped.isEvaluationLimitReached(),
                "cap of one is reached");
        double gap = capped.error() - optimum.error();
        check(gap >= -1e-12,
                "capped heuristic cannot beat the exhaustive optimum");
        System.out.println("  [residual capped] search=" + capped.error()
                + ", exhaustive=" + optimum.error() + ", gap=" + gap
                + " (heuristic, not claimed globally optimal)");
    }

    @Test
    public void returnedSelectionAlwaysSatisfiesBudget() {
        buildFixtures();
        long[] budgets = {0L, 5L, 6L, 7L, 10L, 11L, 12L, 100L, 1000L};
        for (long budget : budgets) {
            for (long seed = 1; seed <= 4; seed++) {
                SearchResult result =
                        new PruningSearcher(chainNet, chainData)
                                .search(budget, seed, 1000);
                if (result.isFeasible()) {
                    check(result.totalRetainedParams() <= budget,
                            "chain params " + result.totalRetainedParams()
                                    + " must respect budget " + budget);
                    check(result.totalRetainedParams() >= 0,
                            "parameter count non-negative");
                }
                SearchResult residualResult =
                        new PruningSearcher(residualNet, residualData)
                                .search(budget, seed, 1000);
                if (residualResult.isFeasible()) {
                    check(residualResult.totalRetainedParams() <= budget,
                            "residual params must respect budget "
                                    + budget);
                }
            }
        }
    }

    @Test
    public void budgetExactlyAtSchemeBoundaryIsAccepted() {
        buildFixtures();
        SearchResult exact = new PruningSearcher(chainNet, chainData)
                .search(12L, 1, 1000);
        check(exact.isFeasible(), "budget exactly equal to params fits");
        checkEquals(12L, exact.totalRetainedParams(),
                "exact-boundary parameter count");

        SearchResult exactPruned = new PruningSearcher(chainNet, chainData)
                .search(7L, 1, 1000);
        check(exactPruned.isFeasible(),
                "budget exactly equal to the only fitting scheme fits");
        check(!new PruningSearcher(chainNet, chainData)
                        .search(6L, 1, 1000).isFeasible(),
                "budget one below the minimum is infeasible");
        checkEquals(7L, exactPruned.totalRetainedParams(),
                "tight exact-boundary parameter count");
    }

    // ------------------------------------------------------------------
    // Tests: seeded reproducibility
    // ------------------------------------------------------------------

    @Test
    public void fixedSeedReproducesSameEvaluationSequence() {
        buildFixtures();
        long[] seeds = {1L, 2L, 7L, 42L};
        for (long seed : seeds) {
            PruningSearcher.SearchRun first =
                    new PruningSearcher(seedNet, seedData)
                            .run(100, seed, 4);
            PruningSearcher.SearchRun second =
                    new PruningSearcher(seedNet, seedData)
                            .run(100, seed, 4);
            check(first.result().evaluations()
                            == second.result().evaluations(),
                    "repeated run evaluation counts match for seed " + seed);
            check(first.result().evaluations() >= 2,
                    "cap allows at least the anchor plus one coarse draw");
            check(first.evaluationTrace().size()
                            == second.evaluationTrace().size(),
                    "trace sizes match");
            for (int step = 0;
                    step < first.evaluationTrace().size();
                    step++) {
                check(Arrays.equals(first.evaluationTrace().get(step),
                                second.evaluationTrace().get(step)),
                        "trace step " + step + " identical for seed "
                                + seed);
            }
            check(first.result().isFeasible()
                            == second.result().isFeasible()
                    && first.result().error()
                            == second.result().error(),
                    "repeated run best error identical for seed " + seed);
        }

        java.util.Set<String> secondSteps = new java.util.HashSet<>();
        for (long seed = 1; seed <= 40; seed++) {
            PruningSearcher.SearchRun run =
                    new PruningSearcher(seedNet, seedData)
                            .run(100, seed, 4);
            secondSteps.add(Arrays.toString(run.evaluationTrace().get(1)));
        }
        check(secondSteps.size() >= 2,
                "different seeds must drive different search paths, got "
                        + secondSteps.size() + " distinct second steps");
    }

    // ------------------------------------------------------------------
    // Tests: evaluation allowance
    // ------------------------------------------------------------------

    @Test
    public void evaluationLimitIsStrictAndReturnsBestSoFar() {
        buildFixtures();
        for (int limit = 0; limit <= 3; limit++) {
            SearchResult result = new PruningSearcher(manualNet, manualData)
                    .search(100, 1, limit);
            checkEquals(Math.min(limit, 2), result.evaluations(),
                    "manual net evaluations for limit " + limit);
            check(result.isEvaluationLimitReached() || limit > 2,
                    "limit reached for exhausted allowance at limit "
                            + limit);
        }

        SearchResult zero = new PruningSearcher(manualNet, manualData)
                .search(100, 1, 0);
        check(!zero.isFeasible(), "no evaluations means no feasible result");
        checkEquals(0, zero.evaluations(), "zero evaluations spent");
        check(zero.toString().contains("no feasible"),
                "zero-limit result reports no feasible scheme explicitly");

        SearchResult one = new PruningSearcher(manualNet, manualData)
                .search(100, 1, 1);
        check(one.isFeasible() && one.isEvaluationLimitReached(),
                "limit 1 returns best-so-far and marks the limit");
        checkClose(2.7125, one.error(), 1e-12,
                "best-so-far is the anchor error");
    }

    @Test
    public void duplicateCandidatesAreNotChargedTwice() {
        buildFixtures();
        PruningSearcher.SearchRun run =
                new PruningSearcher(dedupNet, dedupData)
                        .run(100, 1, 1000);
        checkEquals(3, run.result().evaluations(),
                "four candidates but one is a tensor duplicate: only three "
                        + "distinct schemes evaluated");
        checkEquals(3, run.evaluationTrace().size(),
                "trace lists one entry per actual evaluation");
        ExhaustiveReference.Optimum optimum =
                ExhaustiveReference.optimize(dedupNet, dedupData, 100);
        checkEquals(3, optimum.feasibleEvaluations(),
                "exhaustive reference also evaluates three distinct tensors");
        checkClose(optimum.error(), run.result().error(), 1e-12,
                "dedup search reaches exhaustive optimum");
    }

    @Test
    public void equalErrorsResolveByStableLayerSequence() {
        buildFixtures();
        SearchResult orderA = new PruningSearcher(tieNetA, tieData)
                .search(100, 1, 1000);
        SearchResult orderB = new PruningSearcher(tieNetB, tieData)
                .search(100, 1, 1000);
        checkClose(orderA.error(), orderB.error(), 0.0,
                "both orderings contain the same two equal-error tensors");
        check(orderA.error() == orderB.error(),
                "tied error must compare exactly equal");
        check(Arrays.equals(new int[]{0}, orderA.selection())
                        && Arrays.equals(new int[]{0}, orderB.selection()),
                "lexicographically smallest layer sequence always wins, "
                        + "independent of proposal order");
    }

    // ------------------------------------------------------------------
    // Tests: illegal inputs and non-composable configurations
    // ------------------------------------------------------------------

    @Test
    public void rejectsEmptyCandidateLayer() {
        buildFixtures();
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new LayerSpec(Activation.RELU, null,
                        new ArrayList<LayerCandidate>());
            }
        }, "empty candidate list");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new NetworkDefinition(2,
                        Collections.<LayerSpec>emptyList());
            }
        }, "empty layer list");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new NetworkDefinition(0,
                        Collections.singletonList(
                                new LayerSpec(Activation.RELU, null,
                                        Collections.singletonList(
                                                candidate(
                                                        new double[][]{
                                                                {1.0, 1.0}},
                                                        new double[]{0.0},
                                                        0)))));
            }
        }, "non-positive input dimension");
    }

    @Test
    public void rejectsWrongDimensionsAndNonFiniteMatrices() {
        buildFixtures();
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                candidate(new double[][]{{1.0, 2.0}, {3.0}},
                        new double[]{0.0, 0.0}, 0);
            }
        }, "ragged weight matrix");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                candidate(new double[][]{{1.0, Double.NaN}},
                        new double[]{0.0}, 0);
            }
        }, "non-finite weight");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                candidate(new double[][]{{1.0, 2.0}},
                        new double[]{Double.POSITIVE_INFINITY}, 0);
            }
        }, "non-finite bias");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new LayerCandidate(new double[][]{{1.0, 2.0}},
                        new double[]{0.0}, -1, 0);
            }
        }, "negative retained params");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new LayerCandidate(new double[][]{{1.0, 2.0}},
                        new double[]{0.0}, 3, -1);
            }
        }, "negative pruning level");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new LayerCandidate(new double[][]{{1.0, 2.0}},
                        new double[]{0.0}, 2, 0);
            }
        }, "retained params not equal to tensor entries");
    }

    @Test
    public void rejectsNonComposableLayerDimensions() {
        buildFixtures();
        final LayerCandidate first = candidate(
                new double[][]{{1.0, 0.0}, {0.0, 1.0}},
                new double[]{0.0, 0.0}, 0);
        final LayerCandidate secondWrongIn = candidate(
                new double[][]{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}},
                new double[]{0.0, 0.0}, 0);
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new NetworkDefinition(2, Arrays.asList(
                        new LayerSpec(Activation.IDENTITY, null,
                                Collections.singletonList(first)),
                        new LayerSpec(Activation.IDENTITY, null,
                                Collections.singletonList(
                                        secondWrongIn))));
            }
        }, "adjacent layers with incompatible widths");
    }

    @Test
    public void rejectsNonComposableResidualConfiguration() {
        buildFixtures();
        final LayerCandidate l0 = candidate(
                new double[][]{{1.0, 0.0}, {0.0, 1.0}},
                new double[]{0.0, 0.0}, 0);
        final LayerCandidate l1WrongResidual = candidate(
                new double[][]{{1.0, 0.0, 0.0},
                        {0.0, 1.0, 0.0},
                        {0.0, 0.0, 1.0}},
                new double[]{0.0, 0.0, 0.0}, 0);
        final LayerCandidate l2 = candidate(
                new double[][]{{1.0, 0.0, 0.0},
                        {0.0, 1.0, 0.0},
                        {0.0, 0.0, 1.0}},
                new double[]{0.0, 0.0, 0.0}, 0);
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new NetworkDefinition(2, Arrays.asList(
                        new LayerSpec(Activation.IDENTITY, null,
                                Collections.singletonList(l0)),
                        new LayerSpec(Activation.IDENTITY, 0,
                                Collections.singletonList(
                                        l1WrongResidual)),
                        new LayerSpec(Activation.IDENTITY, null,
                                Collections.singletonList(l2))));
            }
        }, "residual source/target widths can never align");

        final LayerCandidate only = candidate(
                new double[][]{{1.0, 0.0}, {0.0, 1.0}},
                new double[]{0.0, 0.0}, 0);
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new NetworkDefinition(2, Arrays.asList(
                        new LayerSpec(Activation.IDENTITY, null,
                                Collections.singletonList(only)),
                        new LayerSpec(Activation.IDENTITY, 5,
                                Collections.singletonList(only))));
            }
        }, "residual reference to a non-existing/forward layer");
    }

    @Test
    public void concreteSelectionCompositionFailuresAreReported() {
        buildFixtures();
        checkThrows(CompositionException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new ComposedNetwork(chainNet, new int[]{0, 1});
            }
        }, "narrow then wide selection cannot be composed");
        checkThrows(CompositionException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new ComposedNetwork(chainNet, new int[]{0});
            }
        }, "selection length mismatch");
        checkThrows(CompositionException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new ComposedNetwork(chainNet, new int[]{0, 9});
            }
        }, "candidate index out of range");
        checkThrows(CompositionException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new ComposedNetwork(residualNet, new int[]{1, 0, 0});
            }
        }, "residual width mismatch at runtime selection");
    }

    @Test
    public void rejectsBadCalibrationData() {
        buildFixtures();
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new CalibrationData(new ArrayList<double[]>(),
                        new ArrayList<double[]>());
            }
        }, "empty calibration set");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new PruningSearcher(manualNet,
                        new CalibrationData(
                                Collections.singletonList(new double[]{1.0}),
                                Collections.singletonList(
                                        new double[]{1.0, 2.0})));
            }
        }, "calibration input width differs from the network input width");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new CalibrationData(
                        Arrays.asList(new double[]{1.0},
                                new double[]{1.0, 2.0}),
                        Arrays.asList(new double[]{1.0},
                                new double[]{1.0, 2.0}));
            }
        }, "inconsistent calibration vector lengths");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new CalibrationData(
                        Collections.singletonList(
                                new double[]{Double.NaN}),
                        Collections.singletonList(new double[]{1.0}));
            }
        }, "non-finite calibration input");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                new PruningSearcher(manualNet,
                        new CalibrationData(
                                Arrays.asList(new double[]{1.0, 2.0},
                                        new double[]{1.0, 2.0}),
                                Arrays.asList(new double[]{3.0, 4.0, 5.0},
                                        new double[]{3.0, 4.0, 5.0})));
            }
        }, "calibration reference width differs from network output width");
    }

    @Test
    public void rejectsInvalidSearchArguments() {
        buildFixtures();
        final PruningSearcher searcher =
                new PruningSearcher(manualNet, manualData);
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                searcher.search(-1, 1, 10);
            }
        }, "negative budget");
        checkThrows(IllegalArgumentException.class, new ThrowingRunnable() {
            @Override
            public void run() {
                searcher.search(10, 1, -1);
            }
        }, "negative evaluation limit");
    }

    @Test
    public void noFeasibleSchemeIsReportedNotOptimality() {
        buildFixtures();
        SearchResult result = new PruningSearcher(chainNet, chainData)
                .search(0L, 1, 1000);
        check(!result.isFeasible(),
                "zero budget with non-empty matrices is infeasible");
        check(result.toString().toLowerCase().contains("no feasible"),
                "infeasible outcome must state no feasible scheme");
        check(!result.toString().toLowerCase().contains("globally optimal")
                        || result.toString().toLowerCase()
                                .contains("not"),
                "heuristic output must never be labelled globally optimal");

        SearchResult capped = new PruningSearcher(residualNet, residualData)
                .search(32L, 1, 1);
        check(capped.toString().contains("heuristic"),
                "capped result is labelled heuristic");
        check(capped.toString().contains("not claimed to be globally "
                + "optimal"), "capped result disclaims global optimality");
    }
}
