from scipy.sparse import csr_array
from numba import jit
import numpy as np

def compute_sims(M, sims, num_items):
    for i in range(num_items):
        print("Upto row", i)
        for j in range(i, num_items):
            print("Col", j)
            if i == j:
                sims[i, j] = 1
            else:
                item_1 = M[:, [i]]
                item_2 = M[:, [j]]
                sims[i, j] = cos_sim(item_1, item_2)

def cos_sim(item_1, item_2):
    users_1, _ = item_1.nonzero()
    users_2, _ = item_2.nonzero()
    common_users = np.intersect1d(users_1, users_2)

    # Numerator
    a = 0
    b = 0
    c = 0
    for user in common_users:
        r_1 = item_1[user, 0] - 1
        r_2 = item_2[user, 0] - 1
        a += (r_1)*(r_2)
        b += r_1**2
        c += r_2**2

    b = np.sqrt(b)
    c = np.sqrt(c)
    sim = a/(b*c) if a*c != 0 else 0

    return sim

@jit(nopython=True)
def top_n(n, sims, user_prefs, num_items, k):
    _, items = user_prefs.nonzero()
    top = []
    for i in range(num_items):
        # Get neighbours
        neighbs = []
        for j in items:
            sim = sims[i, j] if j >= i else sims[j, i]
            neighbs.append((sim, j))
        neighbs.sort(key=lambda x: x[0], reverse=True)
        neighbs = neighbs[:k]

        # Compute rating
        a = 0
        b = 0
        for sim, j in neighbs:
            r = user_prefs[0, j] - 1
            a += sim*r
            b += sim

        pred = a/b if b != 0 else 0
        top.append((pred, i))
    top.sort(key=lambda x: x[0], reverse=True)
    top = top[:n]

    return top
        
class ItemKNN:
    def __init__(self, k=5):
        self._k = k

    def fit(self, M):
        self._M = M
        self._num_users, self._num_items = M.shape
        self._sims = csr_array((self._num_items, self._num_items))
        compute_sims(M, self._sims, self._num_items)

    def top_n(self, user, n):
        prefs = self._M[user, :]
        top = top_n(n, self._sims, prefs, self._num_items, self._k)
        return top

        


    