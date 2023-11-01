package com.dof.nn.core;

import com.dof.nn.matrix.Matrix;

import java.util.LinkedList;

public class BatchResult {
    private LinkedList<Matrix> io = new LinkedList<>();
    private LinkedList<Matrix> weightErrors = new LinkedList<>();
    private LinkedList<Matrix> weightInputs = new LinkedList<>();
    private Matrix inputError;
    private double loss;
    private double percentCorrect;

    public void addWeightInput(Matrix input) {
        weightInputs.add(input);
    }

    public LinkedList<Matrix> getWeightInputs() {
        return weightInputs;
    }

    public LinkedList<Matrix> getIo() {
        return io;
    }

    public Matrix getOutput() {
        return io.getLast();
    }

    public void addIo(Matrix m) {
        io.add(m);
    }

    public LinkedList<Matrix> getWeightErrors() {
        return weightErrors;
    }

    public void addWeightsError(Matrix weightsError) {
        weightErrors.addFirst(weightsError);
    }

    public Matrix getInputError() {
        return inputError;
    }

    public void setInputError(Matrix inputError) {
        this.inputError = inputError;
    }

    public double getLoss() {
        return loss;
    }

    public void setLoss(double loss) {
        this.loss = loss;
    }

    public double getPercentCorrect() {
        return percentCorrect;
    }

    public void setPercentCorrect(double percentCorrect) {
        this.percentCorrect = percentCorrect;
    }
}
