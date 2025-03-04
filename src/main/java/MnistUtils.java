import java.io.IOException;

public class MnistUtils {

    public static void main(String[] args) throws IOException {
        MnistMatrix[] mnistMatrix = new MnistDataReader().readData("src/main/data/train-images.idx3-ubyte",
                                                                "src/main/data/train-labels.idx1-ubyte");
        printMnistMatrix(mnistMatrix[mnistMatrix.length - 1]);
        mnistMatrix = new MnistDataReader().readData("src/main/data/t10k-images.idx3-ubyte",
                                                        "src/main/data/t10k-labels.idx1-ubyte");
        printMnistMatrix(mnistMatrix[0]);

    }
    public static double[][][] trainData(){
        double[][][] data = new double[2][][];
        try {
            MnistMatrix[] mnistMatrix = new MnistDataReader().readData("src/main/data/train-images.idx3-ubyte",
                    "src/main/data/train-labels.idx1-ubyte");
            data[0] = new double[mnistMatrix.length][];
            data[1] = new double[mnistMatrix.length][];
            for (int i = 0; i < mnistMatrix.length; i++) {
                double[] inputData = new double[28 * 28];
                int[][] mxData = mnistMatrix[i].getData();
                for (int k = 0; k < mxData.length; k++) {
                    int[] row = mxData[k];
                    for (int j = 0; j < row.length; j++) {
                        inputData[k * 28 + j] = row[j];
                    }
                }
                data[0][i] = inputData;
                double[] outputData = new double[10];
                outputData[mnistMatrix[i].getLabel()] = 1;
                data[1][i] = outputData;

            }
        }catch (IOException ignored){}
        return data;

    }

    public static double[][][] testData(){
        double[][][] data = new double[2][][];
        try {
            MnistMatrix[]  mnistMatrix = new MnistDataReader().readData("src/main/data/t10k-images.idx3-ubyte",
                    "src/main/data/t10k-labels.idx1-ubyte");
            data[0] = new double[mnistMatrix.length][];
            data[1] = new double[mnistMatrix.length][];
            for (int i = 0; i < mnistMatrix.length; i++) {
                double[] inputData = new double[28 * 28];
                int[][] mxData = mnistMatrix[i].getData();
                for (int k = 0; k < mxData.length; k++) {
                    int[] row = mxData[k];
                    for (int j = 0; j < row.length; j++) {
                        inputData[k * 28 + j] = row[j];
                    }
                }
                data[0][i] = inputData;
                double[] outputData = new double[10];
                outputData[mnistMatrix[i].getLabel()] = 1;
                data[1][i] = outputData;

            }
        }catch (IOException ignored){}
        return data;

    }
    private static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }
}
