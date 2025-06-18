import torch
from torch.autograd import grad

# -----------------------------------------------
# This script demonstrates vector and tensor calculus
# using PyTorch by computing partial derivatives (gradients)
# of scalar functions with up to 3 variables.
# Two scalar functions are used:
#   1. Polynomial: f(x, y, z) = x^2 + 3*y - z
#   2. Trigonometric: f(x, y, z) = sin(x) + cos(y) + tan(z)
# -----------------------------------------------

# ---- Define a polynomial scalar function ----
def polynomial_function(x):
    # Returns: f(x, y, z) = x^2 + 3*y - z
    return x[0]**2 + 3*x[1] - x[2]

# ---- Define a trigonometric scalar function ----
def trigonometric_function(x):
    # Returns: f(x, y, z) = sin(x) + cos(y) + tan(z)
    return torch.sin(x[0]) + torch.cos(x[1]) + torch.tan(x[2])

# ---- Define input tensor of 3 variables (x, y, z) ----
# Enable autograd by setting requires_grad=True
x_values = torch.tensor([1.0, 0.5, 2.0], requires_grad=True)

# ---- Evaluate both functions at the input values ----
f_polynomial = polynomial_function(x_values)
f_trigonometric = trigonometric_function(x_values)

print("Polynomial function value:", f_polynomial.item())
print("Trigonometric function value:", f_trigonometric.item())

# ---- Compute gradients (partial derivatives) ----
# Each result is a tuple containing a tensor of shape (3,)
# We extract the first (and only) element to get the actual gradient vector
grad_polynomial = grad(f_polynomial, x_values, create_graph=True)[0]
grad_trigonometric = grad(f_trigonometric, x_values, create_graph=True)[0]

# ---- Print partial derivatives of polynomial function ----
print("\nPartial derivatives of polynomial function:")
for i in range(3):
    # grad_polynomial[i] is ∂f/∂xi for i = 0,1,2
    print(f"∂f/∂x{i+1} =", grad_polynomial[i].item())

# ---- Print partial derivatives of trigonometric function ----
print("\nPartial derivatives of trigonometric function:")
for i in range(3):
    # grad_trigonometric[i] is ∂f/∂xi for i = 0,1,2
    print(f"∂f/∂x{i+1} =", grad_trigonometric[i].item())
