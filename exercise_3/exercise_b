import torch

# ------------------------------------------------------
# This script demonstrates fundamental vector and matrix
# operations using PyTorch, including:
#   - Scalar (dot) product
#   - Vector (cross) product
#   - Matrix-vector and matrix-matrix products
#   - Transpose, inverse, trace, determinant, eigenvalues
#   - With shape and dimensionality printed at every step
# ------------------------------------------------------

# ---- Define two 1D vectors with 3 elements ----
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])

# Print shape and dimension of vectors
print("v1 =", v1)
print("Shape of v1:", v1.shape, "| Dimensions:", v1.dim())

print("v2 =", v2)
print("Shape of v2:", v2.shape, "| Dimensions:", v2.dim())

# ---- Scalar (Dot) Product ----
dot_product = torch.dot(v1, v2)
print("\nDot product (v1 · v2):", dot_product.item())
print("Shape of dot product:", dot_product.shape, "| Dimensions:", dot_product.dim())  # Should be scalar (0D)

# ---- Vector (Cross) Product ----
cross_product = torch.linalg.cross(v1, v2)
print("\nCross product (v1 × v2):", cross_product)
print("Shape:", cross_product.shape, "| Dimensions:", cross_product.dim())

# ---- Define a 3×3 matrix A ----
A = torch.tensor([
    [1.0, 2.0, 3.0],
    [0.0, 1.0, 4.0],
    [5.0, 6.0, 0.0]
])
print("\nMatrix A:\n", A)
print("Shape of A:", A.shape, "| Dimensions:", A.dim())

# ---- Matrix-vector multiplication: A · v1 ----
Av = torch.matmul(A, v1)
print("\nMatrix-vector product (A · v1):", Av)
print("Shape:", Av.shape, "| Dimensions:", Av.dim())

# ---- Define another 3×3 matrix B ----
B = torch.tensor([
    [7.0, 8.0, 9.0],
    [0.0, 1.0, 2.0],
    [3.0, 4.0, 5.0]
])
print("\nMatrix B:\n", B)
print("Shape of B:", B.shape, "| Dimensions:", B.dim())

# ---- Matrix-matrix multiplication: A · B ----
AB = torch.matmul(A, B)
print("\nMatrix-matrix product (A · B):\n", AB)
print("Shape:", AB.shape, "| Dimensions:", AB.dim())

# ---- Transpose of A ----
A_T = A.T
print("\nTranspose of A (Aᵀ):\n", A_T)
print("Shape:", A_T.shape, "| Dimensions:", A_T.dim())

# ---- Inverse of a 2×2 matrix C (must be non-singular) ----
C = torch.tensor([
    [4.0, 7.0],
    [2.0, 6.0]
])
print("\nMatrix C:\n", C)
print("Shape of C:", C.shape, "| Dimensions:", C.dim())

C_inv = torch.inverse(C)
print("Inverse of C:\n", C_inv)
print("Shape:", C_inv.shape, "| Dimensions:", C_inv.dim())

# ---- Trace of C ----
trace_C = torch.trace(C)
print("\nTrace of C:", trace_C.item())
print("Shape:", trace_C.shape, "| Dimensions:", trace_C.dim())  # Scalar

# ---- Determinant of C ----
det_C = torch.det(C)
print("Determinant of C:", det_C.item())
print("Shape:", det_C.shape, "| Dimensions:", det_C.dim())  # Scalar

# ---- Eigenvalues and Eigenvectors of a symmetric matrix D ----
D = torch.tensor([
    [2.0, -1.0],
    [-1.0, 2.0]
])
print("\nMatrix D (symmetric):\n", D)
print("Shape:", D.shape, "| Dimensions:", D.dim())

eigenvalues, eigenvectors = torch.linalg.eig(D)
print("Eigenvalues of D:", eigenvalues)
print("Shape:", eigenvalues.shape, "| Dimensions:", eigenvalues.dim())

print("Eigenvectors of D (columns):\n", eigenvectors)
print("Shape:", eigenvectors.shape, "| Dimensions:", eigenvectors.dim())
