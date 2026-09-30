package pruner;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Calibration inputs and reference outputs used to score composed networks.
 */
public final class CalibrationData {

    private final List<double[]> inputs;
    private final List<double[]> referenceOutputs;

    public CalibrationData(List<double[]> inputs,
                           List<double[]> referenceOutputs) {
        if (inputs == null || inputs.isEmpty()) {
            throw new IllegalArgumentException(
                    "at least one calibration sample is required");
        }
        if (referenceOutputs == null
                || referenceOutputs.size() != inputs.size()) {
            throw new IllegalArgumentException(
                    "inputs and reference outputs must have the same "
                            + "non-empty size");
        }
        int inputDim = inputs.get(0).length;
        int outputDim = referenceOutputs.get(0).length;
        if (inputDim <= 0 || outputDim <= 0) {
            throw new IllegalArgumentException(
                    "calibration vectors must not be empty");
        }

        List<double[]> inputCopy = new ArrayList<>(inputs.size());
        List<double[]> outputCopy =
                new ArrayList<>(referenceOutputs.size());
        for (int sample = 0; sample < inputs.size(); sample++) {
            double[] input = inputs.get(sample);
            double[] output = referenceOutputs.get(sample);
            if (input == null || output == null
                    || input.length != inputDim
                    || output.length != outputDim) {
                throw new IllegalArgumentException(
                        "all calibration inputs must have the same length and "
                                + "all reference outputs the same length");
            }
            for (double value : input) {
                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                            "calibration input contains a non-finite value");
                }
            }
            for (double value : output) {
                throwIfNonFiniteOutput(value);
            }
            inputCopy.add(input.clone());
            outputCopy.add(output.clone());
        }
        this.inputs = Collections.unmodifiableList(inputCopy);
        this.referenceOutputs = Collections.unmodifiableList(outputCopy);
    }

    private static void throwIfNonFiniteOutput(double value) {
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException(
                    "reference output contains a non-finite value");
        }
    }

    public List<double[]> inputs() {
        return inputs;
    }

    public List<double[]> referenceOutputs() {
        return referenceOutputs;
    }

    public int size() {
        return inputs.size();
    }
}
