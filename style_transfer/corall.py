import numpy as np


'''def corall(src, dst):
    "Correlation Alignment"
    src_flat = src.reshape(-1, 3)
    src_flat_mean = np.mean(src_flat, 0, keepdims=True)
    src_flat_std = np.std(src_flat, 0, keepdims=True)
    src_flat_norm = (src_flat - src_flat_mean) / src_flat_std
    src_flat_cov_eye = np.matmul(src_flat_norm.T, src_flat_norm) + np.eye(3)

    dst_flat = dst.reshape(-1, 3)
    dst_flat_mean = np.mean(dst_flat, 0, keepdims=True)
    dst_flat_std = np.std(dst_flat, 0, keepdims=True)
    dst_flat_norm = (dst_flat - dst_flat_mean) / dst_flat_std
    dst_flat_cov_eye = np.matmul(dst_flat_norm.T, dst_flat_norm) + np.eye(3)

    src_flat_norm_transfer = np.matmul(src_flat_norm, np.matmul(
        np.linalg.inv(_mat_sqrt(src_flat_cov_eye)),
        _mat_sqrt(dst_flat_cov_eye)
    ))
    src_flat_transfer = src_flat_norm_transfer * dst_flat_std + dst_flat_mean
    return src_flat_transfer.reshape(src.shape)


def _mat_sqrt(m):
    u, s, v = np.linalg.svd(m)
    return np.matmul(np.matmul(u, np.diag(np.sqrt(s))), v)'''

def matSqrt_numpy(x):
    U,D,V = np.linalg.svd(x)
    result = U.dot(np.diag(np.sqrt(D))).dot(V.T)
    return result

def corall(source, target):
    n_channels = source.shape[-1]

    source = np.moveaxis(source, -1, 0)  # HxWxC -> CxHxW
    target = np.moveaxis(target, -1, 0)  # HxWxC -> CxHxW

    source_flatten = source.reshape(n_channels, source.shape[1]*source.shape[2])
    target_flatten = target.reshape(n_channels, target.shape[1]*target.shape[2])

    source_flatten_mean = source_flatten.mean(axis=1, keepdims=True)
    source_flatten_std = source_flatten.std(axis=1, keepdims=True)
    source_flatten_norm = (source_flatten - source_flatten_mean) / source_flatten_std

    target_flatten_mean = target_flatten.mean(axis=1, keepdims=True)
    target_flatten_std = target_flatten.std(axis=1, keepdims=True)
    target_flatten_norm = (target_flatten - target_flatten_mean) / target_flatten_std

    source_flatten_cov_eye = source_flatten_norm.dot(source_flatten_norm.T) + np.eye(n_channels)
    target_flatten_cov_eye = target_flatten_norm.dot(target_flatten_norm.T) + np.eye(n_channels)

    source_flatten_norm_transfer = matSqrt_numpy(target_flatten_cov_eye).dot(np.linalg.inv(matSqrt_numpy(source_flatten_cov_eye))).dot(source_flatten_norm)
    source_flatten_transfer = source_flatten_norm_transfer * target_flatten_std + target_flatten_mean

    coraled = source_flatten_transfer.reshape(source.shape)
    coraled = np.moveaxis(coraled, 0, -1)  # CxHxW -> HxWxC
    return coraled
