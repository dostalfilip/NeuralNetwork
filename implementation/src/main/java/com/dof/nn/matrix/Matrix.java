package com.dof.nn.matrix;

import java.util.Arrays;

public class Matrix {

    private static final String NUMBER_FORMAT = "%+12.5f";
    private static final double TOLERANCE = 0.000001;

    private int rows;
    private int cols;

    public interface Producer {
        double produce(int index);
    }

    public interface IndexValueProducer {
        double produce(int index, double value);
    }

    public interface ValueProducer {
        double produce(double value);
    }

    public interface IndexValueConsumer {
        void consume(int index, double value);
    }

    public interface RowColValueConsumer {
        void consume(int row, int col, double value);
    }

    public interface RowColIndexValueConsumer {
        void consume(int row, int col, int index, double value);
    }

    public interface RowColProducer {
        double produce(int row, int col, double value);
    }

    private double[] a;

    private Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        a = new double[rows * cols];
    }

    public  int getRows() {
        return rows;
    }

    public  int getCols() {
        return cols;
    }

    public Matrix apply(IndexValueProducer producer) {
        Matrix result = new Matrix(rows, cols);

        for (int i = 0; i < a.length; i++) {
            result.a[i] = producer.produce(i, a[i]);
        }

        return result;
    }

    public Matrix modify(ValueProducer producer){
        for (int i = 0; i < a.length; ++i) {
            a[i] = producer.produce(a[i]);
        }
        return this;
    }

    public void forEach(RowColIndexValueConsumer consumer) {
        int index = 0;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                consumer.consume(row, col, index, a[index++]);
            }
        }
    }

    public void forEach(RowColValueConsumer consumer) {
        int index = 0;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                consumer.consume(row, col, a[index++]);
            }
        }
    }

    public void forEach(IndexValueConsumer consumer) {
        for (int i = 0; i < a.length; ++i) {
            consumer.consume(i, a[i]);
        }
    }

    public Matrix modify(RowColProducer producer) {
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                a[row * cols + col] = producer.produce(row, col, a[row * cols + col]);
            }
        }
        return this;
    }

    public Matrix(int rows, int cols, Producer producer) {
        this(rows, cols);

        for (int i = 0; i < a.length; i++) {
            a[i] = producer.produce(i);
        }
    }

    public Matrix sumColumns(){
        Matrix result = new Matrix(1, cols);

        int index = 0;

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                result.a[col] += a[index++];
            }
        }

        return result;
    }

    public Matrix softMax() {
        Matrix result = new Matrix(rows, cols, i -> Math.exp(a[i]));

        Matrix colSum = result.sumColumns();

        result.modify((row, col, value) -> value / colSum.get(col));

        return result;
    }

    public void set(int row, int col, double value){
        a[row * cols + col] = value;
    }

    public double get(int row, int col) {
        return a[row * cols + col];
    }

    public Matrix addIncrement(int row, int col, double increment) {
        Matrix result = apply((index, value) ->  a[index]);

        double originalValue = get(row, col);

        double newValue = originalValue + increment;

        result.set(row, col, newValue);

        return result;
    }

    public double get(int index) {
        return a[index];
    }

    public Matrix multiply(Matrix m) {
        Matrix result = new Matrix(rows, m.cols);

        if (cols != m.rows) {
            throw new RuntimeException("Cannot multiply; wrong number of rows vs cols");
        }

        /* Speed test - for loop order 1000 row/cols multiplication - 20x run sum
            row col n -> 663ms
            row n col -> 634ms
            col n row -> 1240ms
            col row n -> 729ms
            n row col -> 690ms
            n col row -> 1499ms
         */
        for (int row = 0; row < result.rows; row++) {
            for (int n = 0; n < cols; n++) {
                for (int col = 0; col < result.cols; col++) {
                    result.a[row * result.cols + col] += a[row * cols + n] * m.a[col + n * m.cols];
                }
            }
        }

        return result;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        Matrix other = (Matrix) o;

        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - other.a[i]) > TOLERANCE) {
                return false;
            }
        }

        return true;
    }

    @Override
    public int hashCode() {
        int result = rows;
        result = 31 * result + cols;
        result = 31 * result + Arrays.hashCode(a);
        return result;
    }

    public String toString(boolean showValues){
        if (showValues){
            return toString();
        }
        else {
            return rows + "x" + cols;
        }
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();

        int index = 0;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                sb.append(String.format(NUMBER_FORMAT, a[index]));

                index++;
            }
            sb.append("\n");
        }

        return sb.toString();
    }

}
