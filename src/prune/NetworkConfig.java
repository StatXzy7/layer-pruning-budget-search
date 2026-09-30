package prune;

import java.util.Arrays;

/**
 * A small feed-forward residual network.
 *
 * Layers are executed in order. Layer {@code i} computes
 * {@code z = activation(W_i * a_{i-1} + b_i)}; when layer i declares a
 * residual skip from layer j, {@code z += a_j}. The input to layer 0 is
 * the network input, so a residual requires j < i and equal output
 * dimensions of layers j and i. Input dimension of layer i must equal the
 * output dimension of the previous layer (the residual source does not
 * change the propagated dimension).
 *
 * All weight matrices are stored in row-major form {@code W[out][in]}.
 */
public final class NetworkConfig {
    private final LayerSpec[] layers;
    private final double[][][] weights;
    private final double[][] biases;

    private NetworkConfig(LayerSpec[] layers, double[][][] weights, double[][] biases) {
        this.layers = layers;
        this.weights = weights;
        this.biases = biases;
    }

    /** Validates layers / weights / biases and returns an immutable config. */
    public static NetworkConfig create(LayerSpec[] layers, double[][][] weights, double[][] biases) {
        if (layers == null || weights == null || biases == null) {
            throw new InvalidProblemException("layers, weights and biases must all be non-null");
        }
        if (layers.length == 0) {
            throw new InvalidProblemException("network must contain at least one layer");
        }
        if (weights.length != layers.length || biases.length != layers.length) {
            throw new InvalidProblemException("number of weight matrices/bias vectors must match layers");
        }
        for (int i = 0; i < layers.length; i++) {
            if (layers[i] == null) {
                throw new InvalidProblemException("layer " + i + " is null");
            }
            if (weights[i] == null || biases[i] == null) {
                throw new InvalidProblemException("weights/biases of layer " + i + " are null");
            }
            int out = layers[i].outDim();
            int in = layers[i].inDim();
            if (weights[i].length != out) {
                throw new InvalidProblemException("layer " + i + " weight rows " + weights[i].length
                        + " != outDim " + out);
            }
            if (biases[i].length != out) {
                throw new InvalidProblemException("layer " + i + " bias length " + biases[i].length
                        + " != outDim " + out);
            }
            for (int r = 0; r < out; r++) {
                if (weights[i][r] == null || weights[i][r].length != in) {
                    throw new InvalidProblemException("layer " + i + " weight row " + r
                            + " must have length inDim " + in);
                }
                for (int c = 0; c < in; c++) {
                    if (!Double.isFinite(weights[i][r][c])) {
                        throw new InvalidProblemException("non-finite weight at layer " + i);
                    }
                }
                if (!Double.isFinite(biases[i][r])) {
                    throw new InvalidProblemException("non-finite bias at layer " + i);
                }
            }
            if (i > 0 && layers[i].inDim() != layers[i - 1].outDim()) {
                throw new InvalidProblemException("layer " + i + " inDim does not compose with previous outDim");
            }
            int src = layers[i].residualFrom();
            if (src == -1) {
                continue;
            }
            if (src < 0 || src >= i) {
                throw new InvalidProblemException("layer " + i
                        + " residual source must point to an earlier layer, got " + src);
            }
            if (layers[src].outDim() != out) {
                throw new InvalidProblemException("layer " + i + " residual from layer " + src
                        + " has mismatched output dimensions (" + out + " vs " + layers[src].outDim() + ")");
            }
        }
        LayerSpec[] ls = layers.clone();
        double[][][] w = new double[weights.length][][];
        double[][] b = new double[biases.length][];
        for (int i = 0; i < weights.length; i++) {
            w[i] = new double[weights[i].length][];
            for (int r = 0; r < weights[i].length; r++) {
                w[i][r] = weights[i][r].clone();
            }
            b[i] = biases[i].clone();
        }
        return new NetworkConfig(ls, w, b);
    }

    public int layerCount() {
        return layers.length;
    }

    public LayerSpec layer(int i) {
        return layers[i];
    }

    public int inputDim() {
        return layers[0].inDim();
    }

    public int outputDim() {
        return layers[layers.length - 1].outDim();
    }

    /** Defensive copy of layer i weight matrix. */
    public double[][] weights(int i) {
        double[][] out = new double[weights[i].length][];
        for (int r = 0; r < out.length; r++) {
            out[r] = weights[i][r].clone();
        }
        return out;
    }

    public double[] biases(int i) {
        return biases[i].clone();
    }

    /**
     * Returns a new config that uses the supplied replacement weights
     * (one matrix per layer, matching each layer's declared dimensions).
     * Biases are taken from this network.
     */
    public NetworkConfig withWeights(double[][][] replacement) {
        if (replacement == null) {
            throw new InvalidProblemException("replacement weights must not be null");
        }
        if (replacement.length != layers.length) {
            throw new InvalidProblemException("replacement must contain one matrix per layer");
        }
        for (int i = 0; i < layers.length; i++) {
            if (replacement[i] == null || replacement[i].length != weights[i].length) {
                throw new InvalidProblemException("replacement matrix for layer " + i
                        + " has wrong output dimension");
            }
            for (int r = 0; r < replacement[i].length; r++) {
                if (replacement[i][r] == null
                        || replacement[i][r].length != weights[i][r].length) {
                    throw new InvalidProblemException("replacement matrix for layer " + i
                            + " has wrong input dimension");
                }
                for (int c = 0; c < replacement[i][r].length; c++) {
                    if (!Double.isFinite(replacement[i][r][c])) {
                        throw new InvalidProblemException("non-finite replacement weight at layer " + i);
                    }
                }
            }
        }
        return create(layers, replacement, biases);
    }

    double[][][] internalWeightsCopy() {
        double[][][] out = new double[weights.length][][];
        for (int i = 0; i < weights.length; i++) {
            out[i] = weights(i);
        }
        return out;
    }

    @Override
    public String toString() {
        return "NetworkConfig[layers=" + layers.length
                + ", input=" + inputDim() + ", output=" + outputDim() + "]";
    }
}
