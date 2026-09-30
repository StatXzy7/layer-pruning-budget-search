package pruner;

/**
 * Scores a composed network by genuinely executing it on every calibration
 * sample. The error is the mean squared difference over all samples and all
 * output units; it is an end-to-end error, never a sum of per-layer local
 * errors.
 */
public final class Scorer {

    private final CalibrationData data;

    public Scorer(CalibrationData data) {
        if (data == null) {
            throw new IllegalArgumentException(
                    "calibration data must not be null");
        }
        this.data = data;
    }

    public double meanSquaredError(ComposedNetwork network) {
        if (network == null) {
            throw new IllegalArgumentException(
                    "network must not be null");
        }
        double sum = 0.0;
        long count = 0L;
        for (int sample = 0; sample < data.size(); sample++) {
            double[] predicted =
                    network.predict(data.inputs().get(sample));
            double[] reference = data.referenceOutputs().get(sample);
            if (predicted.length != reference.length) {
                throw new IllegalStateException(
                        "network output length " + predicted.length
                                + " does not match reference length "
                                + reference.length);
            }
            for (int unit = 0; unit < predicted.length; unit++) {
                double difference = predicted[unit] - reference[unit];
                sum += difference * difference;
            }
            count += predicted.length;
        }
        return sum / count;
    }
}
