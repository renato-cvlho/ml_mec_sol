import torch

# -----------------------------------------------------------------------------
# This script demonstrates how to define displacement fields in 1D, 2D, and 3D,
# and compute corresponding strain tensors and stress tensors using PyTorch.
# The goal is to simulate small-deformation theory from solid mechanics.
# -----------------------------------------------------------------------------

# --------------------------- 1D DISPLACEMENT FIELD ---------------------------
# Displacement u(x) = 0.01 * x^2
# Compute the 1D strain: ε = du/dx
x = torch.tensor(2.0, requires_grad=True)  # 1D point
u = 0.01 * x**2  # displacement field
strain_1d = torch.autograd.grad(u, x)[0]  # first derivative is strain

print("--- 1D Case ---")
print(f"Displacement u(x): {u.item()}")
print(f"Strain ε(x): {strain_1d.item()}")

# --------------------------- 2D DISPLACEMENT FIELD ---------------------------
# Displacement vector u(x, y) = [0.01*x*y, 0.02*y^2]
# Compute the linearized 2D strain tensor (symmetric gradient of displacement)
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# u1 = 0.01*x*y, u2 = 0.02*y**2
u1 = 0.01 * x * y
u2 = 0.02 * y**2

du1_dx, du1_dy = torch.autograd.grad(u1, (x, y), create_graph=True)
du2_dx, du2_dy = torch.autograd.grad(u2, (x, y), create_graph=True, allow_unused=True)
if du2_dx is None:
    du2_dx = torch.tensor(0.0)

# Build the symmetric strain tensor:
# ε = 0.5 * (∇u + (∇u)^T)
# Components:
# ε_xx = ∂u1/∂x
# ε_yy = ∂u2/∂y
# ε_xy = 0.5 * (∂u1/∂y + ∂u2/∂x)
strain_2d = torch.stack([
    torch.stack([du1_dx, 0.5 * (du1_dy + du2_dx)]),
    torch.stack([0.5 * (du1_dy + du2_dx), du2_dy])
])

print("\n--- 2D Case ---")
print("Strain tensor ε(x, y):")
print(strain_2d)

# --------------------------- 3D DISPLACEMENT FIELD ---------------------------
# u(x, y, z) = [a*y*z, b*x*z, c*x*y]
a, b, c = 0.01, 0.02, 0.03
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = torch.tensor(3.0, requires_grad=True)

u1 = a * y * z
u2 = b * x * z
u3 = c * x * y

du1 = torch.autograd.grad(u1, (x, y, z), create_graph=True, allow_unused=True)
du2 = torch.autograd.grad(u2, (x, y, z), create_graph=True, allow_unused=True)
du3 = torch.autograd.grad(u3, (x, y, z), create_graph=True, allow_unused=True)

# Replace None gradients with zeros for safety
du1 = [torch.tensor(0.0) if g is None else g for g in du1]
du2 = [torch.tensor(0.0) if g is None else g for g in du2]
du3 = [torch.tensor(0.0) if g is None else g for g in du3]

# Build 3x3 strain tensor using symmetric gradient
strain_3d = torch.zeros((3, 3))
for i, dui in enumerate([du1, du2, du3]):
    for j in range(3):
        strain_3d[i, j] += 0.5 * (dui[j])
        strain_3d[j, i] += 0.5 * (dui[j])

print("\n--- 3D Case ---")
print("Strain tensor ε(x, y, z):")
print(strain_3d)

# ---------------------- MATERIAL AND STRESS TENSOR ---------------------------
# Now compute stress from strain using Hooke's law (linear elasticity - plane strain condition)
# σ = C : ε where C is the fourth-order elasticity tensor.
# In isotropic materials: σ_ij = λ * δ_ij * tr(ε) + 2μ * ε_ij
# We'll apply this to the 2D strain tensor above.

E = 210e9  # Young's modulus (Pa)
nu = 0.3   # Poisson's ratio

# Lame parameters
lambda_lame = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))

trace_strain_2d = strain_2d[0, 0] + strain_2d[1, 1]
identity_2d = torch.eye(2)

# Compute stress tensor: σ = λ * tr(ε) * I + 2μ * ε
stress_2d = lambda_lame * trace_strain_2d * identity_2d + 2 * mu * strain_2d

print("\n--- Stress Tensor (2D Linear Elasticity) ---")
print("Stress tensor σ(x, y):")
print(stress_2d)
