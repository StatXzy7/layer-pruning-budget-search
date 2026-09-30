package pruner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Description of one feed-forward layer: activation, optional residual source
 * and the ordered list of pruning candidates (index {@code 0} is the first
 * candidate and also the tie-breaking order used by the search).
 *
 * <p>A residual connection adds the residual source activation to the
 * post-affine values before the activation is applied. The residual source is
 * either the network input ({@code -1}) or the activation output of an earlier
 * layer ({@code 0 .. layerIndex-1}). Residual addition requires equal widths.
 */
public final class LayerSpec {

    private final Activation activation;
    private final Integer residualFrom;
    private final List<LayerCandidate> candidates;

    public LayerSpec(Activation activation, Integer residualFrom,
                     List<LayerCandidate> candidates) {
        if (activation == null) {
            throw new IllegalArgumentException(
                    "activation must not be null");
        }
        if (candidates == null || candidates.isEmpty()) {
            throw new IllegalArgumentException(
                    "each layer must provide at least one pruning candidate");
        }
        List<LayerCandidate> copy = new ArrayList<>(candidates.size());
        for (LayerCandidate candidate : candidates) {
            if (candidate == null) {
                throw new IllegalArgumentException(
                        "candidate must not be null");
            }
            copy.add(candidate);
        }
        this.activation = activation;
        this.residualFrom = residualFrom;
        this.candidates = Collections.unmodifiableList(copy);
    }

    public Activation activation() {
        return activation;
    }

    /** {@code null} means no residual; {@code -1} means the network input. */
    public Integer residualFrom() {
        return residualFrom;
    }

    public List<LayerCandidate> candidates() {
        return candidates;
    }
}
