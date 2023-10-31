package com.dof.nn.core;

import com.dof.nn.matrix.Matrix;

public class TrainingMatrices {
    private Matrix input;
    private Matrix output;

    public TrainingMatrices(Matrix input, Matrix output) {
        this.input = input;
        this.output = output;
    }

    public Matrix getInput() {
        return input;
    }

    public Matrix getOutput() {
        return output;
    }
}
