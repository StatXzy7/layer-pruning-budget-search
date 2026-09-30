package prune;

/**
 * Executes a composed network end-to-end and scores calibration error.
 * Pure Java standard library only; all work is on the CPU.
 */
public final class NetworkRunner {
    private NetworkRunner() {
    }

    /** Runs one input vector through the network and returns its output. */
    public static double[] run(NetworkConfig net, double[] input) {
        if (net == null || input == null) {
            throw new InvalidProblemException("network and input must not be null");
        }
        if (input.length != net.inputDim()) {
            throw new InvalidProblemException("input length " + input.length
                    + " != network input dim " + net.inputDim());
        }
        int layers = net.layerCount();
        double[][] activations = new double[layers][];
        double[] current = input;
        for (int i = 0; i < layers; i++) {
            LayerSpec spec = net.layer(i);
            double[][] w = net.weights(i);
            double[] b = net.biases(i);
            double[] z = new double[spec.outDim()];
            for (int r = 0; r < z.length; r++) {
                double acc = b[r];
                for (int c = 0; c < current.length; c++) {
                    acc += w[r][c] * current[c];
                }
                z[r] = spec.activation().f(acc);
            }
            int src = spec.residualFrom();
            if (src != -1) {
                double[] skip = activations[src];
                for (int r = 0; r < z.length; r++) {
                    z[r] += skip[r];
                }
            }
            activations[i] = z;
            current = z;
        }
        return current;
    }

    /**
     * Mean squared error over all calibration samples and output
     * dimensions. Non-finite produced outputs score as
     * {@code Double.POSITIVE_INFINITY}.
     */
    public static double meanSquaredError(NetworkConfig net,
                                          double[][] inputs,
                                          double[][] references) {
        if (net == null || inputs == null || references == null) {
            throw new InvalidProblemException("net, inputs and references must not be null");
        }
        if (inputs.length != references.length || inputs.length == 0) {
            throw new InvalidProblemException("input/reference arrays must be equally non-empty");
        }
        double sum = 0.0;
        long count = 0;
        for (int s = 0; s < inputs.length; s++) {
            double[] out = run(net, inputs[s]);
            double[] ref = references[s];
            if (ref == null || ref.length != out.length) {
                throw new InvalidProblemException("reference shape mismatch at sample " + s);
            }
            for (int j = 0; j < out.length; j++) {
                double d = out[j] - ref[j];
                double sq = d * d;
                sum += Double.isFinite(sq) ? sq : Double.POSITIVE_INFINITY;
                count++;
            }
        }
        return sum / count;
    }
}
