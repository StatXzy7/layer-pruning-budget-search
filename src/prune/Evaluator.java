package prune;

/**
 * Scores one concrete per-layer candidate combination by actually
 * executing the composed network. Implementations must be deterministic
 * for identical composed weight matrices.
 */
public interface Evaluator {
    double evaluate(double[][][] composedWeights);
}
