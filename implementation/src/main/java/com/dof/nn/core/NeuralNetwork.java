package com.dof.nn.core;

public class NeuralNetwork {
    private Engine engine;

    public NeuralNetwork() {
        this.engine = new Engine();
    }

    public void add(Transform transform, double... params) {
        engine.add(transform, params);
    }

    @Override
    public String toString() {
        return engine.toString();
    }
}
