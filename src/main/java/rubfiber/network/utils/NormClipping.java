package rubfiber.network.utils;

import java.lang.Math;
import java.util.Arrays;

public class NormClipping {

    // Method to calculate the L2 norm of a Double[] vector
    public static double calculateL2Norm(Double[] vector) {
        double sumOfSquares = 0.0;
        for (Double element : vector) {
            // Check for null elements
            if (element != null) {
                sumOfSquares += element * element;
            }
        }
        return Math.sqrt(sumOfSquares);
    }

    // Method to apply L2 norm clipping and return a new Double[] array
    public static Double[] clipByL2Norm(Double[] vector, double maxNorm) {
        if (vector == null) {
            return null;
        }

        double currentNorm = calculateL2Norm(vector);

        if (currentNorm > maxNorm) {
            double ratio = maxNorm / currentNorm;
            Double[] clippedVector = new Double[vector.length];
            for (int i = 0; i < vector.length; i++) {
                if (vector[i] != null) {
                    clippedVector[i] = vector[i] * ratio;
                } else {
                    clippedVector[i] = null;
                }
            }
            return clippedVector;
        } else {
            return Arrays.copyOf(vector, vector.length); // Return a copy
        }
    }

    public static void main(String[] args) {
        // Example with a vector that needs clipping
        Double[] myVector = {3.0, 4.0, 12.0}; // L2 norm is 13.0
        double maxNorm = 10.0;

        double currentNorm = calculateL2Norm(myVector);
        System.out.println("Original vector: " + Arrays.toString(myVector));
        System.out.println("Original L2 norm: " + currentNorm);

        Double[] clippedVector = clipByL2Norm(myVector, maxNorm);
        double newNorm = calculateL2Norm(clippedVector);

        System.out.println("Clipped vector: " + Arrays.toString(clippedVector));
        System.out.println("New L2 norm: " + newNorm);

        // Example with a vector containing a null value
        Double[] vectorWithNull = {3.0, null, 4.0};
        maxNorm = 5.0;
        System.out.println("\nVector with null value: " + Arrays.toString(vectorWithNull));
        System.out.println("Norm before clipping: " + calculateL2Norm(vectorWithNull));
        Double[] clippedWithNull = clipByL2Norm(vectorWithNull, maxNorm);
        System.out.println("Vector after clipping: " + Arrays.toString(clippedWithNull));
    }
}
