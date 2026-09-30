package prune;

/**
 * Thrown whenever the pruning problem specification is invalid: empty
 * candidate layers, bad dimensions, negative parameter counts, non-finite
 * matrices, non-composable residual configuration, malformed calibration
 * data, etc.
 */
public class InvalidProblemException extends RuntimeException {
    private static final long serialVersionUID = 1L;

    public InvalidProblemException(String message) {
        super(message);
    }

    public InvalidProblemException(String message, Throwable cause) {
        super(message, cause);
    }
}
