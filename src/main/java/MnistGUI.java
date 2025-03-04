import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.util.Random;

public class MnistGUI {
    private MnistNeuralNetwork network;
    private JLabel imageLabel;
    private JLabel predictionLabel;

    private Random rand;
    public MnistGUI() {

        network = MnistNeuralNetwork.getInstance();
        JFrame frame = new JFrame("MNIST Neural Network GUI");
        frame.setSize(400, 510);
        frame.getContentPane().setBackground(new Color(251, 234, 235));
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLayout(new BorderLayout());
        frame.setUndecorated(true);

        imageLabel = new JLabel();
        imageLabel.setHorizontalAlignment(JLabel.CENTER);
        frame.add(imageLabel, BorderLayout.CENTER);

        // Create a panel for the buttons and labels
        JPanel panel = new JPanel();
        panel.setLayout(new BorderLayout());
        panel.setBackground(new Color(251, 234, 235));

        // Add a close button
        JButton closeButton = new JButton("Close");
        closeButton.setPreferredSize(new Dimension(400, 50));  // Set size for close button
        closeButton.setBackground(new Color(126, 47, 47));
        closeButton.setForeground(Color.WHITE);
        closeButton.setFocusPainted(false);
        closeButton.setFont(new Font("Tahoma", Font.BOLD, 12));

// Add ActionListener to close the frame when clicked
        closeButton.addActionListener(new AbstractAction() {
            @Override
            public void actionPerformed(ActionEvent e) {
                frame.dispose();
                System.exit(0);
            }
        });  // Close the frame without exiting the app

        panel.add(closeButton, BorderLayout.NORTH );


        JButton loadButton = new JButton("Predict");
        loadButton.setPreferredSize(new Dimension(400, 50));
        loadButton.setBackground(new Color(47, 60, 126));
        loadButton.setForeground(Color.WHITE);
        loadButton.setFocusPainted(false);
        loadButton.setFont(new Font("Tahoma", Font.BOLD, 12));
        loadButton.addActionListener(e -> loadRandomImage());
        panel.add(loadButton);

        predictionLabel = new JLabel("Prediction:     Actual:", SwingConstants.CENTER);
        predictionLabel.setFont(new Font("Tahoma", Font.BOLD, 16));
        panel.add(predictionLabel, BorderLayout.SOUTH   );

        loadRandomImage();

        frame.add(panel, BorderLayout.NORTH);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
    private void loadRandomImage() {
        double[][][] testData = MnistUtils.testData();
        int randomIndex = new Random().nextInt(testData[0].length);
        int label = getLabel(testData[1][randomIndex]);

        BufferedImage bufferedImage = createImageFromData(testData[0][randomIndex]);

        Image scaledImage = bufferedImage.getScaledInstance(395, 395, Image.SCALE_SMOOTH);
        imageLabel.setIcon(new ImageIcon(scaledImage));

        double[] prediction = MnistNeuralNetwork.getInstance().calculate(testData[0][randomIndex]);
        int predictedLabel = getMaxIndex(prediction);
        predictionLabel.setText("Prediction: " + predictedLabel + " Actual: " + label);
    }

    private int getLabel(double[] output) {
        for (int i = 0; i < output.length; i++) {
            if (output[i] == 1) return i;
        }
        return -1;
    }

    private int getMaxIndex(double[] output) {
        int maxIndex = 0;
        for (int i = 1; i < output.length; i++) {
            if (output[i] > output[maxIndex]) {
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    private BufferedImage createImageFromData(double[] data) {
        BufferedImage image = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int value = (int) data[i * 28 + j];
                int gray = 255 - value;
                image.setRGB(j, i, new Color(gray, gray, gray).getRGB());
            }
        }
        return image;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(MnistGUI::new);
    }
}
