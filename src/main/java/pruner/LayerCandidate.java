package pruner;

import java.util.Arrays;

/**
 * One discrete pruning candidate for a single layer.
 *
 * <p>Weights use column-major layout {@code weights[outUnit][inUnit]}: the
 * output unit {@code i} of a layer with {@code inDim} inputs is
 * {@code act(sum_j weights[i][j] * x[j] + biases[i])}. Every candidate has
 * already been pruned and compensated by the caller; this class only stores
 * the resulting tensors together with the true retained-parameter count and
 * the pruning level (larger levels denote coarser pruning, level {@code 0}
 * denotes the unpruned reference).
 */
public final class LayerCandidate {

    private final double[][] weights;
    private final double[] biases;
    private final long retainedParams;
    private final int level;

    public LayerCandidate(double[][] weights, double[] biases,
                          long retainedParams, int level) {
        if (weights == null) {
            throw new IllegalArgumentException("weights must not be null");
        }
        if (weights.length == 0) {
            throw new IllegalArgumentException(
                    "weight matrix must have at least one output unit");
        }
        int inDim = weights[0].length;
        if (inDim == 0) {
            throw new IllegalArgumentException(
                    "weight matrix must have at least one input unit");
        }
        for (double[] row : weights) {
            if (row == null || row.length != inDim) {
                throw new IllegalArgumentException(
                        "weight matrix rows must all have the same length");
            }
            for (double value : row) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                            "weight matrix contains a non-finite value");
                }
            }
        }
        if (biases == null || biases.length != weights.length) {
            throw new IllegalArgumentException(
                    "biases length must match the number of output units");
        }
        for (double value : biases) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                        "bias vector contains a non-finite value");
            }
        }
        if (retainedParams < 0) {
            throw new IllegalArgumentException(
                    "retainedParams must not be negative");
        }
        if (level < 0) {
            throw new IllegalArgumentException(
                    "pruning level must not be negative");
        }
        long tensorParams = (long) weights.length * (long) inDim
                + (long) biases.length;
        if (retainedParams != tensorParams) {
            throw new IllegalArgumentException(
                    "retainedParams (" + retainedParams
                            + ") must equal the number of provided weight and "
                            + "bias entries (" + tensorParams + ")");
        }

        this.weights = new double[weights.length][];
        for (int i = 0; i < weights.length; i++) {
            this.weights[i] = weights[i].clone();
        }
        this.biases = biases.clone();
        this.retainedParams = retainedParams;
        this.level = level;
    }

    public int inDim() {
        return weights[0].length;
    }

    public int outDim() {
        return weights.length;
    }

    public double[][] weights() {
        double[][] copy = new double[weights.length][];
        for (int i = 0; i < weights.length; i++) {
            copy[i] = weights[i].clone();
        }
        return copy;
    }

    public double[] biases() {
        return biases.clone();
    }

    public long retainedParams() {
        return retainedParams;
    }

    public int level() {
        return level;
    }

    /** Structural equality of the pruned tensors (parameter counts excluded). */
    public boolean sameTensorAs(LayerCandidate other) {
        if (other == null
                || weights.length != other.weights.length
                || weights[0].length != other.weights[0].length
                || biases.length != other.biases.length) {
            return false;
        }
        for (int i = 0; i < weights.length; i++) {
            if (!Arrays.equals(weights[i], other.weights[i])) {
                return false;
            }
        }
        return Arrays.equals(biases, other.biases);
    }
}
