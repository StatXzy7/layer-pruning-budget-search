package pruner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * A small feed-forward residual network together with every layer's pruning
 * candidates.
 *
 * <p>Candidate dimensions must be composable: every candidate must take part
 * in at least one end-to-end composable chain. Residual connections must be
 * statically possible on at least one such chain (the residual source and
 * target must admit equal widths with a composable path between them),
 * otherwise the configuration is rejected up front as non-composable.
 */
public final class NetworkDefinition {

    private final int inputDim;
    private final List<LayerSpec> layers;

    public NetworkDefinition(int inputDim, List<LayerSpec> layers) {
        if (inputDim <= 0) {
            throw new IllegalArgumentException("inputDim must be positive");
        }
        if (layers == null || layers.isEmpty()) {
            throw new IllegalArgumentException(
                    "at least one layer is required");
        }
        this.inputDim = inputDim;
        List<LayerSpec> copy = new ArrayList<>(layers.size());
        for (LayerSpec spec : layers) {
            if (spec == null) {
                throw new IllegalArgumentException(
                        "layer spec must not be null");
            }
            copy.add(spec);
        }
        this.layers = Collections.unmodifiableList(copy);
        validate();
    }

    public int inputDim() {
        return inputDim;
    }

    public List<LayerSpec> layers() {
        return layers;
    }

    private void validate() {
        int layerCount = layers.size();

        for (int layer = 0; layer < layerCount; layer++) {
            Integer residualFrom = layers.get(layer).residualFrom();
            if (residualFrom != null && (residualFrom < -1
                    || residualFrom >= layer)) {
                throw new IllegalArgumentException(
                        "layer " + layer + " has an invalid residual source "
                                + residualFrom);
            }
        }

        boolean[][] prefix = new boolean[layerCount][];
        boolean[][] suffix = new boolean[layerCount][];
        for (int layer = 0; layer < layerCount; layer++) {
            int count = layers.get(layer).candidates().size();
            prefix[layer] = new boolean[count];
            suffix[layer] = new boolean[count];
        }

        int firstCount = layers.get(0).candidates().size();
        for (int candidate = 0; candidate < firstCount; candidate++) {
            prefix[0][candidate] =
                    layers.get(0).candidates().get(candidate).inDim()
                            == inputDim;
        }
        for (int layer = 1; layer < layerCount; layer++) {
            List<LayerCandidate> current = layers.get(layer).candidates();
            List<LayerCandidate> previous =
                    layers.get(layer - 1).candidates();
            for (int candidate = 0; candidate < current.size(); candidate++) {
                int neededIn = current.get(candidate).inDim();
                for (int previousCandidate = 0;
                        previousCandidate < previous.size();
                        previousCandidate++) {
                    if (prefix[layer - 1][previousCandidate]
                            && previous.get(previousCandidate).outDim()
                            == neededIn) {
                        prefix[layer][candidate] = true;
                        break;
                    }
                }
            }
        }

        int last = layerCount - 1;
        int lastCount = layers.get(last).candidates().size();
        for (int candidate = 0; candidate < lastCount; candidate++) {
            suffix[last][candidate] = true;
        }
        for (int layer = layerCount - 2; layer >= 0; layer--) {
            List<LayerCandidate> current = layers.get(layer).candidates();
            List<LayerCandidate> next = layers.get(layer + 1).candidates();
            for (int candidate = 0; candidate < current.size();
                    candidate++) {
                int producedOut = current.get(candidate).outDim();
                for (int nextCandidate = 0;
                        nextCandidate < next.size();
                        nextCandidate++) {
                    if (suffix[layer + 1][nextCandidate]
                            && next.get(nextCandidate).inDim()
                            == producedOut) {
                        suffix[layer][candidate] = true;
                        break;
                    }
                }
            }
        }

        for (int layer = 0; layer < layerCount; layer++) {
            int count = layers.get(layer).candidates().size();
            for (int candidate = 0; candidate < count; candidate++) {
                if (!prefix[layer][candidate]
                        || !suffix[layer][candidate]) {
                    throw new IllegalArgumentException(
                            "candidate " + candidate + " of layer " + layer
                                    + " cannot participate in a composable "
                                    + "end-to-end chain (wrong dimensions)");
                }
            }
        }

        for (int layer = 0; layer < layerCount; layer++) {
            Integer residualFrom = layers.get(layer).residualFrom();
            if (residualFrom == null || !residualPossible(
                    layer, residualFrom, prefix, suffix)) {
                if (residualFrom != null) {
                    throw new IllegalArgumentException(
                            "residual connection at layer " + layer
                                    + " from layer " + residualFrom
                                    + " cannot be composed along any "
                                    + "end-to-end chain");
                }
            }
        }
    }

    private boolean residualPossible(int targetLayer, int sourceLayer,
                                     boolean[][] prefix,
                                     boolean[][] suffix) {
        List<LayerCandidate> targetCandidates =
                layers.get(targetLayer).candidates();

        if (sourceLayer == -1) {
            for (int target = 0; target < targetCandidates.size();
                    target++) {
                LayerCandidate targetCandidate = targetCandidates.get(target);
                if (prefix[targetLayer][target]
                        && suffix[targetLayer][target]
                        && targetCandidate.outDim() == inputDim) {
                    return true;
                }
            }
            return false;
        }

        List<LayerCandidate> sourceCandidates =
                layers.get(sourceLayer).candidates();
        for (int source = 0; source < sourceCandidates.size(); source++) {
            if (!prefix[sourceLayer][source]) {
                continue;
            }
            int sourceWidth = sourceCandidates.get(source).outDim();
            for (int target = 0; target < targetCandidates.size();
                    target++) {
                LayerCandidate targetCandidate = targetCandidates.get(target);
                if (!suffix[targetLayer][target]
                        || targetCandidate.outDim() != sourceWidth) {
                    continue;
                }
                if (linkExists(sourceLayer, source, targetLayer, target)) {
                    return true;
                }
            }
        }
        return false;
    }

    private boolean linkExists(int startLayer, int startCandidate,
                               int endLayer, int endCandidate) {
        if (startLayer == endLayer) {
            return startCandidate == endCandidate;
        }
        boolean[] reachable = new boolean[
                layers.get(startLayer).candidates().size()];
        reachable[startCandidate] = true;
        for (int layer = startLayer + 1; layer <= endLayer; layer++) {
            boolean[] nextReachable = new boolean[
                    layers.get(layer).candidates().size()];
            List<LayerCandidate> current = layers.get(layer).candidates();
            List<LayerCandidate> previous =
                    layers.get(layer - 1).candidates();
            for (int candidate = 0; candidate < current.size();
                    candidate++) {
                int neededIn = current.get(candidate).inDim();
                for (int previousCandidate = 0;
                        previousCandidate < previous.size();
                        previousCandidate++) {
                    if (reachable[previousCandidate]
                            && previous.get(previousCandidate).outDim()
                            == neededIn) {
                        nextReachable[candidate] = true;
                        break;
                    }
                }
            }
            reachable = nextReachable;
        }
        return reachable[endCandidate];
    }
}
