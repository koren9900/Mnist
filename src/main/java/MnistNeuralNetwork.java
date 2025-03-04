import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;
import org.apache.commons.math3.linear.RealVectorChangingVisitor;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Scanner;

public class MnistNeuralNetwork {
    private double learning_rate = 0.0001;
    private final double decay_factor = 0.1;
    private final int first_layer = 400;
    private final int second_layer = 80;

    // double beta = 0.9;

    private double acc;

    private double new_acc;
    private RealMatrix first_layer_weights_matrix;
    private RealVector first_layer_bias_vector;
    private RealMatrix second_layer_weights_matrix;
    private RealVector second_layer_bias_vector;
    private RealMatrix third_layer_weights_matrix;
    private RealVector third_layer_bias_vector;

    private static MnistNeuralNetwork instance;
    private MnistNeuralNetwork(){
        acc = readDataAcc();
        readData();
    }
    public static MnistNeuralNetwork getInstance(){
        if(instance == null)
            instance = new MnistNeuralNetwork();
        return instance;
    }

    public void improveAccuracy(){
        if(acc < 90) {
            double[][] first_layer_weights = new double[first_layer][784];
            for (int i = 0; i < first_layer; i++) {
                for (int j = 0; j < 784; j++) {
                    first_layer_weights[i][j] = Utilities.xavierInitializer(784, first_layer);
                }
            }
            first_layer_weights_matrix = MatrixUtils.createRealMatrix(first_layer_weights);

            first_layer_bias_vector = first_layer_bias_vector.map((it) -> Utilities.randomBias());

            double[][] second_layer_weights = new double[second_layer][first_layer];
            for (int i = 0; i < second_layer; i++) {
                for (int j = 0; j < first_layer; j++) {
                    second_layer_weights[i][j] = Utilities.xavierInitializer(first_layer, second_layer);
                }
            }
            second_layer_weights_matrix = MatrixUtils.createRealMatrix(second_layer_weights);

            second_layer_bias_vector = second_layer_bias_vector.map((it) -> Utilities.randomBias());

            double[][] third_layer_weights = new double[10][second_layer];
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < second_layer; j++) {
                    third_layer_weights[i][j] = Utilities.xavierInitializer(second_layer, 10);
                }
            }
            third_layer_weights_matrix = MatrixUtils.createRealMatrix(third_layer_weights);
            third_layer_bias_vector = third_layer_bias_vector.map((it) -> Utilities.randomBias());
        }else {
            readData();
        }
        // Momentum implementation
//        RealMatrix third_layer_velocity = MatrixUtils.createRealMatrix(third_layer_weights_matrix.getRowDimension(), third_layer_weights_matrix.getColumnDimension());
//        RealMatrix second_layer_velocity = MatrixUtils.createRealMatrix(second_layer_weights_matrix.getRowDimension(), second_layer_weights_matrix.getColumnDimension());
//        RealMatrix first_layer_velocity = MatrixUtils.createRealMatrix(first_layer_weights_matrix.getRowDimension(), first_layer_weights_matrix.getColumnDimension());
//
//        RealVector third_layer_bias_velocity = MatrixUtils.createRealVector(new double[third_layer_bias_vector.getDimension()]);
//        RealVector second_layer_bias_velocity = MatrixUtils.createRealVector(new double[second_layer_bias_vector.getDimension()]);
//        RealVector first_layer_bias_velocity = MatrixUtils.createRealVector(new double[first_layer_bias_vector.getDimension()]);


        double[][][] data = MnistUtils.trainData();
        double[][] inputs = data[0];
        double[][] outputs = data[1];
        RealMatrix inputs_matrix= MatrixUtils.createRealMatrix(inputs);
        int epoch = 0;
        while(true){
            for(int i = 0; i < inputs_matrix.getRowDimension(); i++)
                train(inputs[i], outputs[i]);


            learning_rate = learning_rate / (1 + decay_factor * epoch);
            double[][][] test = MnistUtils.testData();
            double[][] test_inputs = test[0];
            double[][] test_outputs = test[1];
            RealMatrix test_inputs_matrix= MatrixUtils.createRealMatrix(test_inputs);
            int count = 0;
            for(int i = 0; i < test_inputs_matrix.getRowDimension(); i++) {
                RealVector input_vector = test_inputs_matrix.getRowVector(i);
                // Calculate output
                // First Layer
                RealVector first_layer_output_vector = first_layer_weights_matrix.operate(input_vector)
                        .add(first_layer_bias_vector);
                first_layer_output_vector.walkInDefaultOrder(new Sigmoid());
                // Second Layer
                RealVector second_layer_output_vector = second_layer_weights_matrix.operate(first_layer_output_vector)
                        .add(second_layer_bias_vector);
                second_layer_output_vector.walkInDefaultOrder(new Sigmoid());
                // Third Layer
                RealVector third_layer_output_vector = third_layer_weights_matrix.operate(second_layer_output_vector)
                        .add(third_layer_bias_vector);
                third_layer_output_vector.walkInDefaultOrder(new Sigmoid());

                third_layer_output_vector = applySoftmax(third_layer_output_vector);
                if(third_layer_output_vector.getMaxIndex() != -1)
                    count += test_outputs[i][third_layer_output_vector.getMaxIndex()];
            }
            new_acc = Math.round(((1.*count) / test_inputs.length) * 10000)/100.;
            System.out.println("Accuracy " + new_acc + "%");
            if(new_acc > acc){
                saveData();
                break;
            }
            epoch++;
        }
    }

    public double[] calculate(double[] input) {
        RealVector input_vector = MatrixUtils.createRealVector(input);
        // Calculate output
        // First Layer
        RealVector first_layer_output_vector = first_layer_weights_matrix.operate(input_vector)
                .add(first_layer_bias_vector);
        first_layer_output_vector.walkInDefaultOrder(new Sigmoid());
        // Second Layer
        RealVector second_layer_output_vector = second_layer_weights_matrix.operate(first_layer_output_vector)
                .add(second_layer_bias_vector);
        second_layer_output_vector.walkInDefaultOrder(new Sigmoid());
        // Third Layer
        RealVector third_layer_output_vector = third_layer_weights_matrix.operate(second_layer_output_vector)
                .add(third_layer_bias_vector);
        third_layer_output_vector.walkInDefaultOrder(new Sigmoid());

        third_layer_output_vector = applySoftmax(third_layer_output_vector);
        return third_layer_output_vector.toArray();
    }

    private RealVector applySoftmax(RealVector vector) {
        double expSum = vector.map(Math::exp).getL1Norm();
        return vector.map((i)->Math.exp(i)/expSum);
    }
    public void saveData(String path) {
        try( FileWriter writer = new FileWriter(path)){
            writer.write(Double.toString(new_acc));
            writer.write("\n");
            writer.write(first_layer_weights_matrix.toString());
            writer.write("\n");
            writer.write(first_layer_bias_vector.toString());
            writer.write("\n");
            writer.write(second_layer_weights_matrix.toString());
            writer.write("\n");
            writer.write(second_layer_bias_vector.toString());
            writer.write("\n");
            writer.write(third_layer_weights_matrix.toString());
            writer.write("\n");
            writer.write(third_layer_bias_vector.toString());
            this.acc = new_acc;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
    private void saveData(){
        saveData("mnistnn.txt");
    }
    private double readDataAcc(){
        File f = new File("mnistnn.txt");
        try(Scanner reader = new Scanner(f)) {
            return reader.nextDouble();
        } catch (FileNotFoundException e) {
            return -1;
        }
    }
    public void readData(){
        readData("mnistnn.txt");
    }
    public void readData(String path){

        File f = new File(path);
        try(Scanner reader = new Scanner(f)) {
            reader.nextDouble();
            reader.nextLine();

            first_layer_weights_matrix = stringToMatrix(reader.nextLine());
            first_layer_bias_vector = stringToVector(reader.nextLine());
            second_layer_weights_matrix = stringToMatrix(reader.nextLine());
            second_layer_bias_vector = stringToVector(reader.nextLine());
            third_layer_weights_matrix = stringToMatrix(reader.nextLine());
            third_layer_bias_vector = stringToVector(reader.nextLine());
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
    private RealMatrix stringToMatrix(String str){
        str = str.substring(1, str.length() - 1);

        String[] rows_str = str.split("}");
        double[][] rows = new double[rows_str.length][];
        for(int i = 0; i < rows_str.length; i++){
            String row_str = rows_str[i];
            while(!Character.isDigit(row_str.charAt(0)) || !Character.isDigit(row_str.charAt(1))
                    && row_str.charAt(0) != '-'
                    &&  (!Character.isDigit(row_str.charAt(0)) || row_str.charAt(1) != '.'))
                row_str = row_str.substring(1);
            String[] row_data_string = row_str.split(",");
            rows[i] = new double[row_data_string.length];
            for(int j = 0; j < row_data_string.length; j++)
                rows[i][j] = Double.parseDouble(row_data_string[j]);
        }
        RealMatrix mtrx = MatrixUtils.createRealMatrix(rows.length, rows[0].length);
        for(int i = 0; i < rows.length; i++){
            mtrx.setRow(i, rows[i]);
        }
        return mtrx;
    }

    private RealVector stringToVector(String str){
        str = str.substring(1, str.length() - 1);

        String[] data_string = str.split(";");
        double[] data = new double[data_string.length];
        for(int j = 0; j < data_string.length; j++)
            data[j] = Double.parseDouble(data_string[j]);

        return MatrixUtils.createRealVector(data);
    }


    public boolean compareExpectedWithOutput(int label, double[] output) {
        RealVector output_vector = MatrixUtils.createRealVector(output);
        return output_vector.getMaxIndex() == label;
    }

    public void train(double[] data, double[] expected) {
        RealVector input_vector = MatrixUtils.createRealVector(data);
        // Calculate output
        // First Layer
        RealVector first_layer_output_vector = first_layer_weights_matrix.operate(input_vector)
                .add(first_layer_bias_vector);

        first_layer_output_vector.walkInDefaultOrder(new Sigmoid());

        // Second Layer
        RealVector second_layer_output_vector = second_layer_weights_matrix.operate(first_layer_output_vector)
                .add(second_layer_bias_vector);
        second_layer_output_vector.walkInDefaultOrder(new Sigmoid());

        // Third Layer
        RealVector third_layer_output_vector = third_layer_weights_matrix.operate(second_layer_output_vector)
                .add(third_layer_bias_vector);

        third_layer_output_vector = applySoftmax(third_layer_output_vector);

        // Third Layer backpropagation

        RealVector third_layer_delta = third_layer_output_vector.subtract(MatrixUtils.createRealVector(expected));

//                third_layer_velocity = third_layer_velocity.scalarMultiply(beta)  // Retain past velocity
//                        .subtract(
//                                third_layer_delta.outerProduct(second_layer_output_vector).scalarMultiply(learning_rate)
//                        );
//
//                third_layer_weights_matrix = third_layer_weights_matrix.add(third_layer_velocity);

        third_layer_weights_matrix = third_layer_weights_matrix.subtract(
                third_layer_delta.outerProduct(second_layer_output_vector).scalarMultiply(learning_rate)
        );

//                third_layer_bias_velocity = third_layer_bias_velocity.mapMultiply(beta)
//                        .subtract(third_layer_delta.mapMultiply(learning_rate));
//
//                third_layer_bias_vector = third_layer_bias_vector.add(third_layer_bias_velocity);

        third_layer_bias_vector = third_layer_bias_vector.subtract(third_layer_delta.mapMultiply(learning_rate));

        // Second Layer backpropagation
        second_layer_output_vector.walkInDefaultOrder(new SigmoidDerivative());

        RealVector second_layer_delta  = third_layer_delta;
        second_layer_delta = third_layer_weights_matrix.transpose().operate(second_layer_delta);
        second_layer_delta = second_layer_delta.ebeMultiply(second_layer_output_vector);

//                second_layer_velocity = second_layer_velocity.scalarMultiply(beta)  // Retain past velocity
//                        .subtract(
//                                second_layer_delta.outerProduct(first_layer_output_vector).scalarMultiply(learning_rate)
//                        );
//
//                second_layer_weights_matrix = second_layer_weights_matrix.add(second_layer_velocity);


        second_layer_weights_matrix = second_layer_weights_matrix.subtract(
                second_layer_delta.outerProduct(first_layer_output_vector).scalarMultiply(learning_rate)
        );

//                second_layer_bias_velocity = second_layer_bias_velocity.mapMultiply(beta)
//                        .subtract(second_layer_delta.mapMultiply(learning_rate));
//
//                second_layer_bias_vector = second_layer_bias_vector.add(second_layer_bias_velocity);

        second_layer_bias_vector = second_layer_bias_vector .subtract(second_layer_delta.mapMultiply(learning_rate));
        // First Layer backpropagation
        first_layer_output_vector.walkInDefaultOrder(new SigmoidDerivative());

        RealVector first_layer_delta  = second_layer_delta;

        first_layer_delta = second_layer_weights_matrix.transpose().operate(first_layer_delta);
        first_layer_delta = first_layer_delta.ebeMultiply(first_layer_output_vector);

//                first_layer_velocity = first_layer_velocity.scalarMultiply(beta)  // Retain past velocity
//                        .subtract(
//                                first_layer_delta.outerProduct(input_vector).scalarMultiply(learning_rate)
//                        );
//
//                first_layer_weights_matrix = first_layer_weights_matrix.add(first_layer_velocity);

        first_layer_weights_matrix = first_layer_weights_matrix.subtract(
                first_layer_delta.outerProduct(input_vector).scalarMultiply(learning_rate)
        );

//                first_layer_bias_velocity = first_layer_bias_velocity .mapMultiply(beta)
//                        .subtract(first_layer_delta.mapMultiply(learning_rate));
//
//                first_layer_bias_vector = first_layer_bias_vector.add(first_layer_bias_velocity);

        first_layer_bias_vector = first_layer_bias_vector.subtract(first_layer_delta.mapMultiply(learning_rate));
    }
}
class Sigmoid implements RealVectorChangingVisitor{
    @Override
    public void start(int i, int i1, int i2) {
        // No initialization needed
    }

    @Override
    public double visit(int i, double v) {
        return 1.0 / (1.0 + Math.exp(-v));
    }

    @Override
    public double end() {
            // No termination needed
            return 0;
    }
}
class SigmoidDerivative implements RealVectorChangingVisitor{
    @Override
    public void start(int i, int i1, int i2) {
        // No initialization needed
    }

    @Override
    public double visit(int i, double v) {
        return v*(1-v);
    }

    @Override
    public double end() {
        // No termination needed
        return 0;
    }
}