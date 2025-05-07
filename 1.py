import numpy as np

A = np.array([
    [ 4, -1,  0, -1,  0,  0],
    [-1,  4, -1,  0, -1,  0],
    [ 0, -1,  4,  0,  1, -1],
    [-1,  0,  0,  4, -1, -1],
    [ 0, -1,  0, -1,  4, -1],
    [ 0,  0, -1,  0, -1,  4]
], dtype=float)
b = np.array([0, -1, 9, 4, 8, 6], dtype=float)

# (a) Jacobi
def jacobi(A, b, x0, tol=1e-8, maxiter=1000):
    D = np.diag(np.diag(A))
    R = A - D
    Dinv = np.linalg.inv(D)
    x = x0.copy()
    for k in range(1, maxiter+1):
        x_new = Dinv @ (b - R @ x)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    return x, maxiter

# (b) Gauss–Seidel
def gauss_seidel(A, b, x0, tol=1e-8, maxiter=1000):
    L = np.tril(A)
    U = A - L
    Linv = np.linalg.inv(L)
    x = x0.copy()
    for k in range(1, maxiter+1):
        x_new = Linv @ (b - U @ x)
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    return x, maxiter

# (c) SOR
def sor(A, b, x0, omega=1.25, tol=1e-8, maxiter=1000):
    n = len(b)
    x = x0.copy()
    for k in range(1, maxiter+1):
        x_old = x.copy()
        for i in range(n):
            sigma = A[i,:i] @ x[:i] + A[i,i+1:] @ x_old[i+1:]
            x[i] = (1-omega)*x_old[i] + (omega/A[i,i])*(b[i] - sigma)
        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            return x, k
    return x, maxiter

# (d) 共軛梯度法
def conjugate_gradient(A, b, x0, tol=1e-8, maxiter=1000):
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rs_old = r @ r
    for k in range(1, maxiter+1):
        Ap = A @ p
        alpha = rs_old / (p @ Ap)
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if np.sqrt(rs_new) < tol:
            return x, k
        p = r + (rs_new/rs_old)*p
        rs_old = rs_new
    return x, maxiter

x0 = np.zeros(6)
sol_j, it_j   = jacobi(A, b, x0)
sol_gs, it_gs = gauss_seidel(A, b, x0)
sol_sor, it_sor = sor(A, b, x0, omega=1.25)
sol_cg, it_cg = conjugate_gradient(A, b, x0)

print(f"Jacobi       → 迭代 {it_j:3d} 次，解 x ≈ {sol_j}")
print(f"Gauss–Seidel → 迭代 {it_gs:3d} 次，解 x ≈ {sol_gs}")
print(f"SOR(ω=1.25)  → 迭代 {it_sor:3d} 次，解 x ≈ {sol_sor}")
print(f"Conj. Grad.  → 迭代 {it_cg:3d} 次，解 x ≈ {sol_cg}")
