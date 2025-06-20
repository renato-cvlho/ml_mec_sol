import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------------
# This script demonstrates how to plot scalar and vector fields
# in 1D, 2D, and 3D using PyTorch (for field definitions)
# and Matplotlib (for visualization).
# Each step is described in detail with explanations of the functions used.
# --------------------------------------------------------------------

# ----------------------------- 1D SCALAR FIELD -----------------------------
# Define a scalar function f(x) = sin(x)
def scalar_field_1d(x):
    return torch.sin(x)

# Create 1D input data
dom_x = torch.linspace(0, 2 * torch.pi, 100)  # domain from 0 to 2pi
values_f = scalar_field_1d(dom_x)             # evaluate the scalar field

# Plot the scalar field
plt.figure(figsize=(6, 4))
plt.plot(dom_x, values_f, label='f(x) = sin(x)', color='blue')
plt.title('1D Scalar Field')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ----------------------------- 2D SCALAR FIELD -----------------------------
# Define scalar field f(x, y) = sin(x) * cos(y)
def scalar_field_2d(x, y):
    return torch.sin(x) * torch.cos(y)

# Create a 2D grid for x and y
x_vals = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100)
y_vals = torch.linspace(-2 * torch.pi, 2 * torch.pi, 100)
X, Y = torch.meshgrid(x_vals, y_vals, indexing='xy')
Z = scalar_field_2d(X, Y)

# Plot 2D scalar field as a contour plot
plt.figure(figsize=(6, 5))
cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(cp)
plt.title('2D Scalar Field: f(x, y) = sin(x) * cos(y)')
plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()

# ----------------------------- 2D VECTOR FIELD -----------------------------
# Define vector field V(x, y) = [cos(y), sin(x)]
def vector_field_2d(x, y):
    return torch.cos(y), torch.sin(x)

# Sample fewer points to avoid overplotting
x_vals_v = torch.linspace(-2, 2, 20)
y_vals_v = torch.linspace(-2, 2, 20)
Xv, Yv = torch.meshgrid(x_vals_v, y_vals_v, indexing='xy')
U, V = vector_field_2d(Xv, Yv)

# Plot 2D vector field using quiver plot
plt.figure(figsize=(6, 5))
plt.quiver(Xv, Yv, U, V, color='red')
plt.title('2D Vector Field: V(x, y) = [cos(y), sin(x)]')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------- 3D VECTOR FIELD -----------------------------
# Define a simple 3D vector field: V(x, y, z) = [y, -x, z]
def vector_field_3d(x, y, z):
    return y, -x, z

# Create a 3D grid
grid_size = 5
x_vals_3d = torch.linspace(-1, 1, grid_size)
y_vals_3d = torch.linspace(-1, 1, grid_size)
z_vals_3d = torch.linspace(-1, 1, grid_size)
X3, Y3, Z3 = torch.meshgrid(x_vals_3d, y_vals_3d, z_vals_3d, indexing='xy')
U3, V3, W3 = vector_field_3d(X3, Y3, Z3)

# Flatten arrays for 3D quiver plotting
X3_np = X3.numpy().flatten()
Y3_np = Y3.numpy().flatten()
Z3_np = Z3.numpy().flatten()
U3_np = U3.numpy().flatten()
V3_np = V3.numpy().flatten()
W3_np = W3.numpy().flatten()

# Plot 3D vector field using quiver3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X3_np, Y3_np, Z3_np, U3_np, V3_np, W3_np, length=0.1, normalize=True, color='green')
ax.set_title('3D Vector Field: V(x, y, z) = [y, -x, z]')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.tight_layout()
plt.show()

# ----------------------------- Explanation of Commands -----------------------------
# torch.linspace(start, end, steps): creates evenly spaced values from start to end
# torch.meshgrid(x, y, indexing='xy'): generates coordinate matrices from coordinate vectors
# plt.plot(): plots 1D curves
# plt.contourf(): plots 2D filled contour map of scalar fields
# plt.quiver(): plots vector fields as arrows in 2D
# ax.quiver(): does the same in 3D with Axes3D
# plt.colorbar(): adds a color scale next to contour plots
# .numpy(): converts a PyTorch tensor to NumPy array for Matplotlib compatibility
