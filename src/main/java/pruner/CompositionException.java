package pruner;

/**
 * Raised when a concrete per-layer candidate selection cannot be composed:
 * adjacent layer widths do not match, a residual width mismatch occurs or a
 * candidate index is out of range.
 */
public class CompositionException extends RuntimeException {

    private static final long serialVersionUID = 1L;

    public CompositionException(String message) {
        super(message);
    }
}
