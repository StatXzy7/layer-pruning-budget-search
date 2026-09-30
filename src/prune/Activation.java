package prune;

/** Element-wise activation functions supported by feed-forward layers. */
public enum Activation {
    LINEAR {
        @Override public double f(double x) { return x; }
    },
    RELU {
        @Override public double f(double x) { return x > 0.0 ? x : 0.0; }
    },
    TANH {
        @Override public double f(double x) { return Math.tanh(x); }
    },
    SIGMOID {
        @Override public double f(double x) { return 1.0 / (1.0 + Math.exp(-x)); }
    };

    public abstract double f(double x);
}
