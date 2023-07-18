from scipy.sparse import csr_array, lil_array
import numba as nb
from numba import jit
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

@jit(nopython=True)
def compute_sims(rows, data, sims, num_items):
    for i in range(num_items):
        print("Upto row", i)
        for j in range(i, num_items):
            if i == j:
                sims[i, j] = 1
            else:
                
                # item_1 = M[:, [i]].toarray()
                # item_2 = M[:, [j]].toarray()
                sims[i, j] = cos_sim(i, j, rows, data)

@jit(nopython=True)
def cos_sim(i, j, rows, data):
    a = 0
    b = 0
    c = 0
    for l, row in enumerate(rows):
        if i in row and j in row:

            r_1 = data[l][np.where(row == i)[0][0]]
            r_2 =  data[l][np.where(row == j)[0][0]]
            a += (r_1)*(r_2)
            b += r_1**2
            c += r_2**2

    b = np.sqrt(b)
    c = np.sqrt(c)
    sim = a/(b*c) if a*c != 0 else 0

    return sim

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
        M = M.tolil()
        self._num_users, self._num_items = M.shape
        self._sims = np.zeros((self._num_items, self._num_items))

        rows = nb.typed.List([np.array(row) for row in M.rows])
        data = nb.typed.List([np.array(dat) for dat in M.data])
        compute_sims(rows, data, self._sims, self._num_items)

    def top_n(self, user, n):
        prefs = self._M[[user], :]
        top = top_n(n, self._sims, prefs, self._num_items, self._k)
        return top

        


    