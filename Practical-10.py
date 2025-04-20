import numpy as np

# Define the sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights and biases for the network
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Randomly initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden = np.random.randn(1, self.hidden_size)
        self.bias_output = np.random.randn(1, self.output_size)

    def forward(self, X):
        # Perform a forward pass through the network
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        
        return self.final_output
    
    def backward(self, X, y, output, learning_rate):
        # Perform backpropagation and update the weights and biases
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 1000 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        output = self.forward(X)
        return output

# Sample data for finite word classification
# Features: Representing words as one-hot encoded vectors
# Categories: [0, 1] where 0 = "Negative", 1 = "Positive"
X = np.array([
    [1, 0, 0],  # Word: "happy" -> Positive
    [0, 1, 0],  # Word: "sad" -> Negative
    [0, 0, 1],  # Word: "joyful" -> Positive
])

y = np.array([
    [1],  # Positive class
    [0],  # Negative class
    [1],  # Positive class
])

# Initialize the neural network with 3 input neurons (for 3 words), 4 hidden neurons, and 1 output neuron
nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1)

# Train the model with 5000 epochs and a learning rate of 0.1
nn.train(X, y, epochs=5000, learning_rate=0.1)

# Test the neural network with the trained model
test_input = np.array([[1, 0, 0]])  # "happy"
prediction = nn.predict(test_input)
print(f"Prediction for 'happy': {prediction}")

test_input = np.array([[0, 1, 0]])  # "sad"
prediction = nn.predict(test_input)
print(f"Prediction for 'sad': {prediction}")

test_input = np.array([[0, 0, 1]])  # "joyful"
prediction = nn.predict(test_input)
print(f"Prediction for 'joyful': {prediction}")
