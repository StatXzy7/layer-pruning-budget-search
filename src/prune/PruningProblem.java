package prune;

import java.util.List;

/**
 * Immutable specification of a cross-layer pruning budget search:
 * the original network, per-layer pruned candidates, calibration
 * inputs / reference outputs, and the retained-parameter budget.
 */
public final class PruningProblem {
    private final NetworkConfig network;
    private final List<List<Candidate>> layerCandidates;
    private final double[][] calibrationInputs;
    private final double[][] referenceOutputs;
    private final long paramBudget;

    public PruningProblem(NetworkConfig network,
                          List<List<Candidate>> layerCandidates,
                          double[][] calibrationInputs,
                          double[][] referenceOutputs,
                          long paramBudget) {
        if (network == null) {
            throw new InvalidProblemException("network must not be null");
        }
        if (layerCandidates == null) {
            throw new InvalidProblemException("layer candidates must not be null");
        }
        if (calibrationInputs == null || referenceOutputs == null) {
            throw new InvalidProblemException("calibration data must not be null");
        }
        if (paramBudget < 0) {
            throw new IllegalArgumentException("parameter budget must not be negative");
        }

        int layers = network.layerCount();
        if (layerCandidates.size() != layers) {
            throw new InvalidProblemException("one candidate list per layer is required, got "
                    + layerCandidates.size() + " for " + layers + " layers");
        }
        for (int i = 0; i < layers; i++) {
            List<Candidate> list = layerCandidates.get(i);
            if (list == null) {
                throw new InvalidProblemException("candidate list for layer " + i + " is null");
            }
            if (list.isEmpty()) {
                throw new InvalidProblemException("layer " + i + " has no candidates");
            }
            int out = network.layer(i).outDim();
            int in = network.layer(i).inDim();
            for (int k = 0; k < list.size(); k++) {
                Candidate c = list.get(k);
                if (c == null) {
                    throw new InvalidProblemException("candidate " + k + " of layer " + i + " is null");
                }
                double[][] w = c.weights();
                if (w.length != out || w[0].length != in) {
                    throw new InvalidProblemException("candidate " + k + " of layer " + i
                            + " has wrong dimensions " + w.length + "x" + w[0].length
                            + ", expected " + out + "x" + in);
                }
            }
        }

        if (calibrationInputs.length == 0) {
            throw new InvalidProblemException("at least one calibration sample is required");
        }
        int samples = calibrationInputs.length;
        if (referenceOutputs.length != samples) {
            throw new InvalidProblemException("reference outputs count must match calibration inputs");
        }
        int inDim = network.inputDim();
        int outDim = network.outputDim();
        for (int s = 0; s < samples; s++) {
            if (calibrationInputs[s] == null || calibrationInputs[s].length != inDim) {
                throw new InvalidProblemException("calibration input " + s
                        + " must have length " + inDim);
            }
            for (int j = 0; j < inDim; j++) {
                if (!Double.isFinite(calibrationInputs[s][j])) {
                    throw new InvalidProblemException("non-finite calibration input at sample " + s);
                }
            }
            if (referenceOutputs[s] == null || referenceOutputs[s].length != outDim) {
                throw new InvalidProblemException("reference output " + s
                        + " must have length " + outDim);
            }
            for (int j = 0; j < outDim; j++) {
                if (!Double.isFinite(referenceOutputs[s][j])) {
                    throw new InvalidProblemException("non-finite reference output at sample " + s);
                }
            }
        }

        this.network = network;
        // Defensive copy; Candidate instances are immutable.
        java.util.ArrayList<java.util.List<Candidate>> copy =
                new java.util.ArrayList<java.util.List<Candidate>>(layers);
        for (int i = 0; i < layers; i++) {
            copy.add(java.util.Collections.unmodifiableList(
                    new java.util.ArrayList<Candidate>(layerCandidates.get(i))));
        }
        this.layerCandidates = java.util.Collections.unmodifiableList(copy);
        this.calibrationInputs = deepCopy2d(calibrationInputs);
        this.referenceOutputs = deepCopy2d(referenceOutputs);
        this.paramBudget = paramBudget;
    }

    private static double[][] deepCopy2d(double[][] src) {
        double[][] out = new double[src.length][];
        for (int i = 0; i < src.length; i++) {
            out[i] = src[i].clone();
        }
        return out;
    }

    public NetworkConfig network() {
        return network;
    }

    public int layerCount() {
        return network.layerCount();
    }

    public List<Candidate> candidates(int layer) {
        return layerCandidates.get(layer);
    }

    public double[][] calibrationInputs() {
        return deepCopy2d(calibrationInputs);
    }

    public double[][] referenceOutputs() {
        return deepCopy2d(referenceOutputs);
    }

    public long paramBudget() {
        return paramBudget;
    }

    public int sampleCount() {
        return calibrationInputs.length;
    }
}
