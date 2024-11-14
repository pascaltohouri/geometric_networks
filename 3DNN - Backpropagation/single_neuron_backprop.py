# Numpy and Pandas
import numpy as np
import pandas as pd
from numpy.ma.core import concatenate

# SciPy
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

# Load dataset
data = pd.read_csv(r"C:\Users\pasca\OneDrive\2024\2024_10\simple_variational_inference\winequality-red.csv")

# Select only 'citric acid' and 'alcohol' columns as inputs
X = data[['citric acid', 'alcohol']].values  # shape: (N, 2)

# Add a column of ones for the bias term
# ones_column = np.ones((X.shape[0], 1))       # Shape: (N, 1)
# X = np.hstack((X, ones_column))              # Shape: (N, 3)

# Import quality variables, convert to float64
y_true = data["quality"].astype(np.float64).values  # Shape (N,)

# Define hyper-parameters
eta = 0.02
iter = 10000

# Set the node number: in-internal-out, and dimensions.
N_0, N_1, N_2, D = 2, 1, 1, 3

# Initialize the coordinate matrices (N_l x D) with zeros
Z_0 = np.array([[0, 0, 0], [0, 0, 1]])
Z_1 = np.array([[0, 0.5, 0.5]])
Z_2 = np.array([[1, 0, 0.5]])

def pairwise_euclidean_distances(A, B):
    # Check both A and B are numpy arrays
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise TypeError("Both A and B must be NumPy arrays.")

    # Check A and B are 2-dimensional matrices
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("Both A and B must be 2-dimensional matrices.")

    # Check A and B have D dimensions
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"Dimensionality mismatch: A has {A.shape[1]} features, B has {B.shape[1]} features.")

    # Calculate the distance between A and B. Row 1, column 1: For Z_n0,d,0 and Z_n1,d,1, calculate ||z_1,d,1 - z_1,d,0||
    distances = cdist(A, B, metric='euclidean')  # Shape: (N_0, N_1)

    return distances

# Compute pairwise Euclidean distances for weights
w_01 = pairwise_euclidean_distances(Z_0, Z_1)
w_12 = pairwise_euclidean_distances(Z_2, Z_1)

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Activations/propagation
def forward(w_01, w_12, X, b):
    # Transpose w_12 and X for correct matrix multiplication
    h = X @ w_01  # Shape: (N, N_1)
    a = sigmoid(h) # Shape: (N, N_1)
    y_pred = a @ w_12 # Shape: (N, N_2)

    return h, a, y_pred

# Mean Squared Error (MSE) loss and its derivative
def mse_loss(y_true, y_pred):
    return np.mean((1/2)*(y_pred - y_true)**2)

def mse_loss_derivative(y_true, y_pred):
    d = y_pred - y_true # Shape: (N,1)
    N = d.shape[0]
    return (1/N)*d

# Backward propagation
def backward(X, y_true, y_pred, h, a, w_01, w_12):
    # Calculate output layer gradient
    dL_dy_pred = mse_loss_derivative(y_true, y_pred) # Shape: (N, N_2)

    # Calculate activation and final weight gradient
    dL_da = dL_dy_pred @ w_12.T     # Shape: (N, N_1)
    dL_dw12 = a.T @ dL_dy_pred   # Shape: (N_1, N_2)

    # calculate hidden derivatives
    da_dh = a*(1-a) # Shape: (N, N_1)
    dL_dh = da_dh * dL_da # Shape: (N, N_1)

    # Calculate initial weight derivatives
    # dh_dw_01 = X # Shape (N, N_1)
    dL_dw_12 = X.T @ dL_dh # Shape: (N_1, N_2)

    # Calculate the weight derivatives wrt. coordinates dW_01_dZ_1 and dW_12_dZ_1
    dW_01_dZ_1 =        # Shape: (N_1, N_0, D)
    dW_12_dZ_1 =         # Shape: (N_1, N_2, D)

    # Compute the loss derivatives wrt. coordinates dL_dZ_1
    # Calculate the W_01 channel
    dL(W_01)_dZ_1 = np.einsum('ij,ijd->id', dL_dW_01, dW_dZ_1) # Shape: (N_1, D)

    # Calculate the W_12 channel
    dL(W_12)_dZ_1 = np.einsum('ij,ijd->id', dL_dW_12, dW_12_dZ_1) # Shape: (N_1, D)

    # Sum the partial derivatives wrt. coordinates
    dL_dZ_1 = dL_dZ_1
    return dL_dZ_2



def train(self, X, y_true, iter=1000):
    # Training loop
    for epoch in range(iter):
        # Forward propagation
        y_pred = forward(w_12, w_23, X)

        # Compute gradient


        # Apply update
        self.backward(X, y_true, y_hat)

        # compute loss
        loss = mse_loss(y_true, y_pred)

        # Print loss every 10 iterations
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss}")


# Sample usage
if __name__ == "__main__":

    # Test the network
    print("\nPredictions after training:")
    print(nn.forward(X))
