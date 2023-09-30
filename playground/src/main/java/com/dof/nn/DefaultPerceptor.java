package com.dof.nn;

/**
 * Hello Neural Network!
 *
 * INPUT    AND     OR      XOR     NOR     NAND    XNOR
 * 00       0       0       0       1       1       1
 * 01       0       1       1       0       1       0
 * 10       0       1       1       0       1       0
 * 11       1       1       0       0       0       1
 *
 */
public class DefaultPerceptor
{
    public static void main( String[] args ) {

        // input
        double[] x = {0.5, 1.0};

        // weights
        double[] w = {0.5, 0.5};

        // bias
        double b = 1.0;

        // weighted sum
        double z = 0.0;

        for(int i = 0; i < x.length; i++){
            z += x[i] * w[i];
        }

        z += b;

        // activation function
        double a = z > 0.0 ? 1.0 : 0.0;

        System.out.println(a);

    }
}
