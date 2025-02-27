import numpy as np
import jax.numpy as jnp
from scipy.spatial import procrustes

def procrustes_analysis(eigenvectors1, eigenvectors2):
    real1, imag1 = np.real(eigenvectors1), np.imag(eigenvectors1)
    real2, imag2 = np.real(eigenvectors2), np.imag(eigenvectors2)

    real_mtx1, real_mtx2, real_disparity = procrustes(real1, real2)

    imag_mtx1, imag_mtx2, imag_disparity = procrustes(imag1, imag2)

    total_disparity = real_disparity + imag_disparity
    print(f"Disparity: {total_disparity}")

def relative_error(A, B):
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape.")

    numerator = jnp.linalg.norm(A - B, ord='fro')
    denominator = jnp.linalg.norm(A, ord='fro')
    rel_error = numerator / denominator

    print("Relative Error:", rel_error)

def load4d(file_path, nx=48, ny=48, nz=48):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.dat', dtype=np.complex128)
    arr = arr.reshape((nx, ny, nz, 2), order='F')
    #return jnp.array(arr)
    return jnp.array(np.transpose(arr, (3, 0, 1, 2)))

def load4d_real(file_path, nx=48, ny=48, nz=48):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.dat', dtype=np.float64)
    arr = arr.reshape((nx, ny, nz, 2), order='F')
    return jnp.array(np.transpose(arr, (3, 0, 1, 2)))

def load3d_real(file_path, nx=48, ny=48, nz=48):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.dat', dtype=np.float64)
    arr = arr.reshape((nx, ny, nz), order='F')
    return jnp.array(arr)

def load5d_real(file_path, nx=48, ny=48, nz=48):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.dat', dtype=np.float64)
    arr = arr.reshape((nx, ny, nz, 3, 2), order='F')
    #return jnp.array(arr)
    return jnp.array(np.transpose(arr, (4, 3, 0, 1, 2)))

def load5d(file_path, nx=48, ny=48, nz=48, nstate=132):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.dat', dtype=np.complex128)
    arr = arr.reshape((nx, ny, nz, 2, nstate), order='F')
    #return jnp.array(arr)
    return jnp.array(np.transpose(arr, (4, 3, 0, 1, 2)))

def load3d(file_path, nx=48, ny=48, nz=48):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.dat', dtype=np.complex128)
    arr = arr.reshape((nx, ny, nz), order='F')
    return jnp.array(np.transpose(arr, (2, 0, 1)))

def load2d(file_path, n=221184, m=82):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.dat', dtype=np.complex128)
    arr = arr.reshape((n, m), order='F')
    return jnp.array(arr)

def load2d_real(file_path, n=82, m=82):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.dat', dtype=np.float64)
    arr = arr.reshape((n, m), order='F')
    return jnp.array(np.transpose(arr, (1, 0)))

def load2d_int(file_path, n=82, m=82):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.dat', dtype=np.int32)
    arr = arr.reshape((n, m), order='F')
    return jnp.array(np.transpose(arr, (1, 0)))

def load1d_real(file_path, n=48):
    arr = np.fromfile('/mnt/home/gulomer/HFBFFT/Code/' + file_path + '.d', dtype=np.float64)
    return jnp.array(arr)
