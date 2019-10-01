import numpy as np


def neg_log_likelihood(x, p, A, S):
    s, q = x[:S], x[S:]

    ps = p(s, q)

    closeness_to_true_value = np.empty_like(ps)
    closeness_to_true_value[A == 1] = ps[A == 1]
    closeness_to_true_value[A == 0] = np.ones_like(ps[A == 0]) - ps[A == 0]
    closeness_to_true_value = np.maximum(1e-12, closeness_to_true_value)

    ll = np.sum(np.log(closeness_to_true_value))
    return -ll / (s.shape[0] * q.shape[0])


def d_neg_log_likelihood(x, p, dpds, dpdq, A, S):
    s, q = x[:S], x[S:]

    ps = p(s, q)

    denominator = np.empty_like(ps)
    denominator[A == 1] = np.maximum(ps[A == 1], 1e-12)
    denominator[A == 0] = np.minimum(ps[A == 0], 1 - 1e-12) - np.ones_like(ps[A == 0])

    dll_ds = np.sum(dpds(s, q) / denominator, axis=1)
    dll_dq = np.sum(dpdq(s, q) / denominator, axis=0)

    dll = np.concatenate((np.reshape(dll_ds, -1), np.reshape(dll_dq, -1)))
    return -dll / (s.shape[0] * q.shape[0])
