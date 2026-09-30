package prune;

/**
 * Immutable specification of one feed-forward layer:
 * out = activation(W * x + b), optionally with a residual skip
 * (out += output of layer {@code residualFrom}).
 *
 * Residual index validity (range and equal output dimensions) is checked
 * by {@link NetworkConfig} once layer positions are known.
 */
public final class LayerSpec {
    private final String name;
    private final int inDim;
    private final int outDim;
    private final Activation activation;
    private final int residualFrom;

    public LayerSpec(String name, int inDim, int outDim, Activation activation, int residualFrom) {
        if (inDim <= 0 || outDim <= 0) {
            throw new InvalidProblemException("layer dimensions must be positive, got "
                    + inDim + "x" + outDim);
        }
        if (activation == null) {
            throw new InvalidProblemException("activation must not be null");
        }
        this.name = name;
        this.inDim = inDim;
        this.outDim = outDim;
        this.activation = activation;
        this.residualFrom = residualFrom;
    }

    public String name() {
        return name;
    }

    public int inDim() {
        return inDim;
    }

    public int outDim() {
        return outDim;
    }

    public Activation activation() {
        return activation;
    }

    /** Index of the layer whose output is added, or -1 when there is no skip. */
    public int residualFrom() {
        return residualFrom;
    }
}
