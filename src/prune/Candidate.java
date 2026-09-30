package prune;

import java.util.Arrays;

/**
 * One pruning candidate for a single layer: a weight matrix already
 * pruned and compensated, the true number of retained parameters
 * (non-zero weight entries), and a pruning level. Larger levels denote
 * coarser / more aggressive pruning; candidates must keep the original
 * layer dimensions so that any per-layer combination composes.
 */
public final class Candidate {
    private final double[][] weights;
    private final long retainedParams;
    private final int level;

    public Candidate(double[][] weights, long retainedParams, int level) {
        if (weights == null) {
            throw new InvalidProblemException("candidate weights must not be null");
        }
        if (weights.length == 0 || weights[0] == null || weights[0].length == 0) {
            throw new InvalidProblemException("candidate weight matrix must be non-empty");
        }
        int in = weights[0].length;
        for (int r = 0; r < weights.length; r++) {
            if (weights[r] == null || weights[r].length != in) {
                throw new InvalidProblemException("candidate matrix must be rectangular");
            }
            for (int c = 0; c < in; c++) {
                if (!Double.isFinite(weights[r][c])) {
                    throw new InvalidProblemException("candidate matrix contains non-finite value");
                }
            }
        }
        if (retainedParams < 0) {
            throw new InvalidProblemException("retained parameter count must not be negative");
        }
        long capacity = (long) weights.length * in;
        if (retainedParams > capacity) {
            throw new InvalidProblemException("retained parameter count " + retainedParams
                    + " exceeds matrix capacity " + capacity);
        }
        if (level < 0) {
            throw new InvalidProblemException("pruning level must not be negative");
        }
        this.weights = new double[weights.length][];
        for (int r = 0; r < weights.length; r++) {
            this.weights[r] = weights[r].clone();
        }
        this.retainedParams = retainedParams;
        this.level = level;
    }

    /** Defensive copy of the candidate weight matrix. */
    public double[][] weights() {
        double[][] out = new double[weights.length][];
        for (int r = 0; r < weights.length; r++) {
            out[r] = weights[r].clone();
        }
        return out;
    }

    public long retainedParams() {
        return retainedParams;
    }

    public int level() {
        return level;
    }

    @Override
    public String toString() {
        return "Candidate[" + weights.length + "x" + weights[0].length
                + ", params=" + retainedParams + ", level=" + level + "]";
    }
}
