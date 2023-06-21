
# Overview

The goal of this investigation is to create a deep learning model using TensorFlow to forecast whether funding requests submitted by the charitable organization Alphabet Soup would be approved. The objective is to develop a model that can correctly predict, depending on numerous input features, whether a funding application will be approved or denied.

# Report/ Results

### DATA PREPROCESSING

#### Target Variable: 
The target variable for the model is the "IS_SUCCESSFUL" column, which says if the funding application was successful or not.

#### Feature Variables: 
The features for the model include many columns like "APPLICATION_TYPE," "AFFILIATION," "CLASSIFICATION," "USE_CASE," "ORGANIZATION," "STATUS," "INCOME_AMT," "SPECIAL_CONSIDERATIONS," and "ASK_AMT."

#### Variables to be Removed: 
The "EIN" and "NAME" columns were removed from the input data, because they are not targets or features.


### COMPILING, TRAINING, AND EVALUATING THE MODEL

#### Number of Neurons: 
The number of neurons in each layer is as follows:
Input layer: The number of neurons in the input layer is determined by the shape of the input data, specifically X_train_scaled.shape[1].

First hidden layer: There are 8000 neurons in the first hidden layer.
Second hidden layer: There are 300 neurons in the second hidden layer.
Third hidden layer: There are 10 neurons in the third hidden layer.
Output layer: There is 1 neuron in the output layer.

#### Number of Layers: 
The neural network has a total of 4 layers: 1 input layer, 3 hidden layers, and 1 output layer.

#### Activation Functions: 
The activation functions used for each layer are as follows:
First hidden layer: LeakyReLU activation function.
Second hidden layer: ReLU activation function.
Third hidden layer: ReLU activation function.
Output layer: Sigmoid activation function.

#### Target model performance:

Loss: 0.5787
Accuracy: 0.7311

The loss value represents the model's performance in terms of how well it minimized the difference between the predicted and actual values during training. Lower loss values indicate better performance.The accuracy value represents the proportion of correctly predicted samples in the dataset. 

### Steps to optimize model performance 

Increasing model complexity: The model has multiple hidden layers with varying numbers of neurons. Increasing the complexity of the model by adding more layers or neurons can potentially improve its capacity to learn and capture complex patterns in the data.

Activation functions: The model uses different activation functions for each hidden layer, including LeakyReLU and ReLU. Activation functions introduce non-linearity into the model, enabling it to learn non-linear relationships between the input features and the target variable.

Model evaluation: The model's performance is evaluated using the summary function nn.summary(), which provides a summary of the model's architecture, including the number of parameters in each layer. This helps in understanding the model's structure and identifying potential issues such as underfitting or overfitting.

### summary

The Alphabet Soup deep learning model produced encouraging outcomes in forecasting the success of funding applications. The model's performance was enhanced by adopting numerous optimizations, including modifying the amount of neurons, epochs, and learning rate. Nevertheless, despite all of these efforts, the final model only met the target accuracy of 75% with an accuracy of 73%.

### Recommendations

Find the best configuration for your model by experimenting with various hyperparameters, such as learning rate, batch size, and number of neurons. To methodically explore the hyperparameter space and find the ideal combination that maximizes performance, use methods like grid search or random search.
Consider adding more layers or neurons to your model to make it more sophisticated, which will improve your model's ability to recognize and learn intricate patterns in the data. To confirm that the model generalizes well to samples that have not been seen before, watch for overfitting and track the model's performance on validation data.





