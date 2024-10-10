# MNIST-Classification-with-PyTorch
This Python application demonstrates how to create, train, and evaluate a neural network for classifying handwritten digits from the MNIST dataset using PyTorch. The project also showcases how to save and load a trained model.
Features
Load and preprocess the MNIST dataset
Build a simple fully connected neural network
Train the model using backpropagation with the Adam optimizer
Test the model's accuracy on the MNIST test dataset
Save and load a trained PyTorch model
Requirements
Before running the application, make sure to have the following libraries installed:

Python 3.x
torch
torchvision
numpy

Training
When you run the script, it will:

Load and preprocess the MNIST dataset.
Define a fully connected neural network (SimpleNN).
Train the model on the training dataset for a specified number of epochs (5 by default).
Evaluate the model's accuracy on the test dataset after each epoch.
The model will be trained and evaluated, and you will see outputs like:

'''
Train Epoch: 1 [0/60000]  Loss: 2.320015
Train Epoch: 1 [6400/60000]  Loss: 0.645912

Test set: Average loss: 0.0014, Accuracy: 9717/10000 (97.17%)
'''
Saving and Loading the Model
After the model is trained, it is saved to a file named mnist_model.pth. To load the saved model, the script includes the following example:

'''
model_loaded = SimpleNN()
model_loaded.load_state_dict(torch.load("mnist_model.pth"))
This allows you to reuse the trained model without retraining.
'''

Model Architecture
The neural network SimpleNN consists of:

Input layer: 28x28 pixels flattened into a 1D vector (784 inputs)
Hidden layer 1: Fully connected layer with 512 neurons
Hidden layer 2: Fully connected layer with 256 neurons
Output layer: 10 neurons corresponding to the 10 digit classes (0-9)
Optimizer and Loss Function
Optimizer: Adam optimizer (optim.Adam) is used to minimize the loss function by updating the model's parameters.
Loss Function: Cross-entropy loss (nn.CrossEntropyLoss) is used to measure the difference between the predicted and actual labels.
Customization
You can easily modify the following aspects:

Batch Size: Change the batch_size variable to control the number of samples processed at once.
Learning Rate: Adjust the learning_rate variable to change how quickly the model updates during training.
Number of Epochs: Modify the epochs variable to control the number of training iterations.
License
This project is licensed under the MIT License. 