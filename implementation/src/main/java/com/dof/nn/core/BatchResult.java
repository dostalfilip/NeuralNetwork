package com.dof.nn.core;

import com.dof.nn.matrix.Matrix;

import java.util.LinkedList;

public class BatchResult {
    private LinkedList<Matrix> io = new LinkedList<>();
    private LinkedList<Matrix> weightsErrors = new LinkedList<>();
    private  Matrix inputError;

    public  LinkedList<Matrix> getIo(){
        return io;
    }

    public Matrix getOutput(){
        return io.getLast();
    }

    public void addIo(Matrix m){
        io.add(m);
    }

    public LinkedList<Matrix> getWeightsErrors() {
        return weightsErrors;
    }

    public void addWeightsError(Matrix weightsError) {
        weightsErrors.addFirst(weightsError);
    }

    public Matrix getInputError() {
        return inputError;
    }

    public void setInputError(Matrix inputError) {
        this.inputError = inputError;
    }
}
