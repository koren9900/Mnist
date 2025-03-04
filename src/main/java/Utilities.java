import java.util.Random;

public class Utilities {
    private static final Random rand = new Random();
    public static double xavierInitializer(int nIn, int nOut) {
        Random rand = new Random();
        double range = Math.sqrt(6.0 / (nIn + nOut));
        return (rand.nextDouble() * 2 - 1) * range;
    }
    public static double randomBias(){
        double stdDev = 0.01; // Standard deviation
        return  stdDev * rand.nextGaussian();
    }
}

