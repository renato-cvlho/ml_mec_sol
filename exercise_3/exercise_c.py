import torch

# --------------------------------------------------------
# This script demonstrates tensor calculus operations
# using PyTorch's autograd system for scalar and vector fields.
# We compute:
#   - Gradients (of scalar fields)
#   - Divergence (of vector fields)
#   - Curl/Rotational (in 3D vector fields)
#   - Jacobian (of vector fields)
#   - Hessian (of scalar fields)
# Each field is a function of x, y, and z.
# --------------------------------------------------------

# Define the input variables (3D vector: x, y, z) with autograd enabled
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

# ---------------------- SCALAR FIELD ----------------------
# Define a scalar field: f(x, y, z)
def scalar_field(x, y, z):
    return x**2 * y + torch.sin(z) * y**2

# Compute the scalar field value
f = scalar_field(x, y, z)

# ---- Gradient: vector of partial derivatives (∂f/∂x, ∂f/∂y, ∂f/∂z) ----
grads = torch.autograd.grad(f, (x, y, z), create_graph=True)
print("\nGradient of scalar field f(x, y, z):")
for i, var in enumerate(['x', 'y', 'z']):
    print(f"∂f/∂{var} =", grads[i])

# ---- Hessian: matrix of second-order partial derivatives ----
print("\nHessian of scalar field f(x, y, z):")
hessian = torch.zeros(3, 3)
variables = (x, y, z)
for i in range(3):
    for j in range(3):
        if grads[i].requires_grad:
            second_grad = torch.autograd.grad(grads[i], variables[j], retain_graph=True, allow_unused=True)[0]
            if second_grad is None:
                hessian[i, j] = torch.tensor(0.0)
            else:
                hessian[i, j] = second_grad
print(hessian)

# ---------------------- VECTOR FIELD ----------------------
# Define a 3D vector field: V(x, y, z) = [Vx, Vy, Vz]
def vector_field(x, y, z):
    Vx = x * y
    Vy = y * z
    Vz = z * x
    return torch.stack([Vx, Vy, Vz])

# Compute the vector field value
V = vector_field(x, y, z)

# ---- Jacobian: matrix of partial derivatives ∂Vi/∂xj ----
print("\nJacobian of vector field V(x, y, z):")
J = torch.zeros(3, 3)
for i in range(3):
    grads_i = torch.autograd.grad(V[i], (x, y, z), retain_graph=True)
    J[i] = torch.tensor(grads_i)
print(J)

# ---- Divergence: sum of diagonal elements of Jacobian ----
# div(V) = ∂Vx/∂x + ∂Vy/∂y + ∂Vz/∂z
divergence = J[0, 0] + J[1, 1] + J[2, 2]
print("\nDivergence of V:", divergence)

# ---- Curl (Rotational): only defined in 3D vector fields ----
# curl(V) = [∂Vz/∂y - ∂Vy/∂z, ∂Vx/∂z - ∂Vz/∂x, ∂Vy/∂x - ∂Vx/∂y]
curl = torch.stack([
    J[2, 1] - J[1, 2],  # ∂Vz/∂y - ∂Vy/∂z
    J[0, 2] - J[2, 0],  # ∂Vx/∂z - ∂Vz/∂x
    J[1, 0] - J[0, 1]   # ∂Vy/∂x - ∂Vx/∂y
])
print("\nCurl (Rotational) of V:", curl)
