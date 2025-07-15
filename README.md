
# MNIST - Digit Recognizer 
This repository contains a Java implementation of a neural network that recognizes handwritten digits from the MNIST dataset.

## Neural Network
The main brain of this project is the MnistNeuralNetwork class. It builds a 3-layer feedforward network:
Input (784) → Hidden1 (400) → Hidden2 (80) → Output (10)
- Weights are initialized using Xavier initialization, or loaded from a saved mnistnn.txt file.
- Training uses momentum for faster convergence.
- The learning rate decays over time as the network iterates over the training data.
- After each epoch, it evaluates accuracy on the test set. If the new accuracy beats the one recorded in mnistnn.txt, it saves the updated weights (with the accuracy written at the top of the file).

## User Interface
A simple Swing-based GUI lets you see predictions live:
- Displays random test images and predicts their digit when you hit "Predict".
- Tracks and displays simple stats: correct predictions, wrong predictions, and current accuracy.
<img width="996" height="510" alt="Screenshot" src="https://github.com/user-attachments/assets/06fe6884-f025-4292-afe7-ac73e2977565" />
