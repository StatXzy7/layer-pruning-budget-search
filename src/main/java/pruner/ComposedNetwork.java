package pruner;

import java.util.Arrays;

/**
 * A concrete, executable network obtained by selecting exactly one pruning
 * candidate for every layer of a {@link NetworkDefinition}.
 *
 * <p>Every forward pass is a genuine end-to-end execution of the composed
 * pruned tensors; residual streams are added pre-activation exactly as
 * declared by the network configuration.
 */
public final class ComposedNetwork {

    private final NetworkDefinition definition;
    private final LayerCandidate[] selected;
    private final int[] selection;

    public ComposedNetwork(NetworkDefinition definition, int[] selection) {
        if (definition == null) {
            throw new IllegalArgumentException(
                    "network definition must not be null");
        }
        if (selection == null
                || selection.length != definition.layers().size()) {
            throw new CompositionException(
                    "selection must provide exactly one candidate index per "
                            + "layer");
        }
        this.definition = definition;
        this.selection = selection.clone();
        this.selected = new LayerCandidate[selection.length];
        for (int layer = 0; layer < selection.length; layer++) {
            int candidateIndex = selection[layer];
            int candidateCount =
                    definition.layers().get(layer).candidates().size();
            if (candidateIndex < 0 || candidateIndex >= candidateCount) {
                throw new CompositionException(
                        "candidate index " + candidateIndex + " for layer "
                                + layer + " is out of range [0,"
                                + candidateCount + ")");
            }
            selected[layer] =
                    definition.layers().get(layer).candidates()
                            .get(candidateIndex);
        }
        checkDimensions();
    }

    public int[] selection() {
        return selection.clone();
    }

    public long totalRetainedParams() {
        long total = 0L;
        for (LayerCandidate candidate : selected) {
            total += candidate.retainedParams();
        }
        return total;
    }

    private void checkDimensions() {
        if (selected[0].inDim() != definition.inputDim()) {
            throw new CompositionException(
                    "layer 0 candidate expects input width "
                            + selected[0].inDim() + " but network input has "
                            + definition.inputDim());
        }
        for (int layer = 1; layer < selected.length; layer++) {
            if (selected[layer].inDim() != selected[layer - 1].outDim()) {
                throw new CompositionException(
                        "layers " + (layer - 1) + " and " + layer
                                + " width mismatch: "
                                + selected[layer - 1].outDim() + " output vs "
                                + selected[layer].inDim() + " input");
            }
        }
        for (int layer = 0; layer < selected.length; layer++) {
            Integer residualFrom =
                    definition.layers().get(layer).residualFrom();
            if (residualFrom == null) {
                continue;
            }
            int sourceWidth = residualFrom == -1
                    ? definition.inputDim()
                    : selected[residualFrom].outDim();
            if (sourceWidth != selected[layer].outDim()) {
                throw new CompositionException(
                        "residual at layer " + layer + " from layer "
                                + residualFrom + " width mismatch: "
                                + sourceWidth + " vs "
                                + selected[layer].outDim());
            }
        }
    }

    /** Executes the composed network on one input vector. */
    public double[] predict(double[] input) {
        if (input == null) {
            throw new IllegalArgumentException("input must not be null");
        }
        if (input.length != definition.inputDim()) {
            throw new IllegalArgumentException(
                    "input length " + input.length
                            + " does not match network input dimension "
                            + definition.inputDim());
        }
        for (double value : input) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                        "input contains a non-finite value");
            }
        }

        double[][] activations = new double[selected.length][];
        double[] current = input.clone();
        for (int layer = 0; layer < selected.length; layer++) {
            LayerCandidate candidate = selected[layer];
            double[][] weights = candidate.weights();
            double[] biases = candidate.biases();
            double[] preActivation = new double[candidate.outDim()];
            for (int out = 0; out < candidate.outDim(); out++) {
                double sum = biases[out];
                double[] row = weights[out];
                for (int in = 0; in < candidate.inDim(); in++) {
                    sum += row[in] * current[in];
                }
                preActivation[out] = sum;
            }

            Integer residualFrom =
                    definition.layers().get(layer).residualFrom();
            if (residualFrom != null) {
                double[] residual = residualFrom == -1
                        ? input : activations[residualFrom];
                for (int unit = 0; unit < preActivation.length; unit++) {
                    preActivation[unit] += residual[unit];
                }
            }

            Activation activation =
                    definition.layers().get(layer).activation();
            double[] output = new double[preActivation.length];
            for (int unit = 0; unit < output.length; unit++) {
                output[unit] = activation.apply(preActivation[unit]);
            }
            activations[layer] = output;
            current = output;
        }
        return current;
    }

    @Override
    public String toString() {
        return "ComposedNetwork" + Arrays.toString(selection());
    }
}
