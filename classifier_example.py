import numpy as np
import matplotlib.pyplot as plt
from nn import Layer, FeedForwardNN, WeightInitializationOption, WeightInitializer, Activation
from optimizers import RMSProp
from metrics import binary_crossentropy_loss
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# 1. Generate synthetic binary classification data
X_np, y_np = make_moons(n_samples=300, noise=0.2, random_state=42)

# Train/test split
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(X_np, y_np, test_size=0.2, random_state=42)

# Convert to lists for compatibility
X_train = X_train_np.tolist()
y_train = y_train_np.tolist()
X_val = X_val_np.tolist()
y_val = y_val_np.tolist()

# 2. Build Feedforward Neural Network
initializer = WeightInitializer(option=WeightInitializationOption.NORMAL)

model = FeedForwardNN(
    layers=[
        Layer(n_input=2, n_output=16, activation=Activation.RELU, initializer=initializer),
        Layer(n_input=16, n_output=16, activation=Activation.RELU, initializer=initializer),
        Layer(n_input=16, n_output=1, activation=Activation.SIGMOID, initializer=initializer),
    ]
)

# 3. Optimizer, Loss, Training
optimizer = RMSProp(parameters=model.parameters(), lr=0.005, beta=0.9)
loss_fn = binary_crossentropy_loss
epochs = 20
batch_size = 32

model.fit(
    X_train=X_train,
    y_train=y_train,
    optimizer=optimizer,
    loss=loss_fn,
    epochs=epochs,
    batch_size=batch_size,
    metric="accuracy",
    X_val=X_val,
    y_val=y_val
)

# 4. Plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    
    # Prepare grid data
    grid_points = np.c_[xx.ravel(), yy.ravel()].tolist()
    preds = model.forward_batch(grid_points)
    probs = np.array([out[-1].val for out in preds])
    probs = probs.reshape(xx.shape)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, probs, levels=[0, 0.5, 1], cmap="RdBu", alpha=0.9)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu", edgecolors='k')
    plt.title("Binary Classification Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.savefig("decision_boundary.png")
    plt.show()

# Call plot function
plot_decision_boundary(model, X_np, y_np)
