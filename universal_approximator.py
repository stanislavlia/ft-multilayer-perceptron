import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from nn import Layer, FeedForwardNN, WeightInitializationOption, WeightInitializer, Activation
from optimizers import RMSProp, StochasticGradientDescent
from metrics import mean_squared_error_loss
from sklearn.model_selection import train_test_split


def complicated_function(x):
    
    return -(-x**2 + 10 * np.log(x*3) * np.sin(x*2) + x ** 1.5 * np.exp(np.cos(x/2))) / 10


x_values = np.linspace(0.1, 10, 100)

# Calculate the corresponding y values
y_values = complicated_function(x_values)


X_data_reshaped = x_values.reshape(-1, 1) # Reshape x_values to be a 2D array for train_test_split

X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_data_reshaped, y_values, test_size=0.1, random_state=42
)

# Convert to list of lists for X and list for y, as expected by FeedForwardNN
X_train = X_train_np.tolist()
y_train = y_train_np.tolist()

X_val = X_val_np.tolist()
y_val = y_val_np.tolist()


initializer = WeightInitializer(option=WeightInitializationOption.NORMAL)

model = FeedForwardNN(
    layers=[
        Layer(n_input=1,  # Single input feature 'x'
              n_output=200, 
              activation=Activation.TANH,
              initializer=initializer),
        Layer(n_input=200,  # Single input feature 'x'
              n_output=1, 
              activation=Activation.LINEAR,
              initializer=initializer),
              ]
)

# 3. Choose Optimizer and Loss
params = model.parameters()
learning_rate = 0.0001
optimizer =  RMSProp(parameters=params , lr=learning_rate, beta=0.95)
loss_fn = mean_squared_error_loss # Appropriate for regression
epochs = 100 # More epochs might be needed for complex functions
batch_size = 8 # Batch size


model.fit(
    X_train=X_train,
    y_train=y_train,
    optimizer=optimizer,
    loss=loss_fn,
    epochs=epochs,
    batch_size=batch_size,
    metric="mse",
    X_val=X_val,
    y_val=y_val)

x_input = x_values.reshape(-1, 1).tolist()  # Shape: List[List[float]]
output_nodes = model.forward_batch(x_input)  # Returns List[List[Node]]

# Extract the final node value (output) from each sample
y_pred = [out[-1].val for out in output_nodes]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, label='Original Function', color='blue', linewidth=2)
plt.plot(x_values, y_pred, label='NN Approximation', color='red', linestyle='--', linewidth=2)
plt.title('Original Function vs. Neural Network Approximation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
print("Save plot...")
plt.savefig('approximation.png')
plt.show()