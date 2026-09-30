package pruner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Tiny self-contained demo: builds a small feed-forward residual network with
 * coarse and fine per-layer pruning candidates, runs the seeded budget search
 * on CPU and compares it with the exhaustive optimum. Run with
 * {@code java pruner.Demo [budget] [seed] [evaluationLimit]}.
 */
public final class Demo {

    private Demo() {
    }

    public static void main(String[] args) {
        long budget = args.length > 0 ? Long.parseLong(args[0]) : 32L;
        long seed = args.length > 1 ? Long.parseLong(args[1]) : 12345L;
        int evaluationLimit = args.length > 2
                ? Integer.parseInt(args[2]) : 1000;

        NetworkDefinition network = buildNetwork();
        CalibrationData data = buildCalibrationData(network);

        System.out.println("layers=" + network.layers().size()
                + ", inputDim=" + network.inputDim()
                + ", budget=" + budget + ", seed=" + seed
                + ", evaluationLimit=" + evaluationLimit);

        PruningSearcher.SearchRun run = new PruningSearcher(network, data)
                .run(budget, seed, evaluationLimit);
        System.out.println("heuristic: " + run.result());
        System.out.println("actual evaluations charged: "
                + run.result().evaluations());

        ExhaustiveReference.Optimum optimum =
                ExhaustiveReference.optimize(network, data, budget);
        if (optimum.isFeasible()) {
            double gap = run.result().isFeasible()
                    ? run.result().error() - optimum.error()
                    : Double.NaN;
            System.out.println("exhaustive optimum: layers="
                    + Arrays.toString(optimum.selection())
                    + ", params=" + optimum.totalRetainedParams()
                    + ", error=" + optimum.error());
            System.out.println("gap to exhaustive optimum: " + gap);
            System.out.println("(positive/zero gap is expected; the heuristic "
                    + "is never labelled globally optimal)");
        } else {
            System.out.println("exhaustive reference: no feasible scheme at "
                    + "or below budget");
        }
    }

    private static NetworkDefinition buildNetwork() {
        Random random = new Random(2026);
        LayerCandidate l0Coarse = candidate(2, 3, 1, random);
        LayerCandidate l0Fine = candidate(3, 3, 0, random);
        LayerCandidate l1Coarse = candidate(3, 2, 1, random);
        LayerCandidate l1Fine = candidate(3, 3, 0, random);
        LayerCandidate l2Coarse = candidate(2, 3, 1, random);
        LayerCandidate l2Fine = candidate(2, 3, 0, random);
        return new NetworkDefinition(3, Arrays.asList(
                new LayerSpec(Activation.RELU, null,
                        Arrays.asList(l0Coarse, l0Fine)),
                new LayerSpec(Activation.IDENTITY, -1,
                        Arrays.asList(l1Coarse, l1Fine)),
                new LayerSpec(Activation.IDENTITY, null,
                        Arrays.asList(l2Coarse, l2Fine))));
    }

    private static LayerCandidate candidate(int outDim, int inDim, int level,
                                            Random random) {
        double[][] weights = new double[outDim][inDim];
        for (int out = 0; out < outDim; out++) {
            for (int in = 0; in < inDim; in++) {
                weights[out][in] = 0.4 * random.nextGaussian();
            }
        }
        double[] biases = new double[outDim];
        for (int out = 0; out < outDim; out++) {
            biases[out] = 0.1 * random.nextGaussian();
        }
        long params = (long) outDim * inDim + outDim;
        return new LayerCandidate(weights, biases, params, level);
    }

    private static CalibrationData buildCalibrationData(
            NetworkDefinition network) {
        Random random = new Random(77);
        List<double[]> inputs = new ArrayList<>();
        for (int sample = 0; sample < 6; sample++) {
            double[] input = new double[network.inputDim()];
            for (int unit = 0; unit < input.length; unit++) {
                input[unit] = random.nextGaussian();
            }
            inputs.add(input);
        }
        ComposedNetwork dense = new ComposedNetwork(network,
                new int[]{1, 1, 1});
        List<double[]> references = new ArrayList<>();
        for (double[] input : inputs) {
            references.add(dense.predict(input));
        }
        return new CalibrationData(inputs, references);
    }
}
