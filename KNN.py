import numba as nb
from numba import jit
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import random

@jit(nopython=True)
def compute_sims(item_ratings, num_users, num_items, means=None) -> np.Array:
    sims = np.zeros((num_items, num_items))

    for i in range(num_items):
        print("Upto row", i)
        for j in range(i, num_items):
            if i == j:
                sims[i, j] = 1
            else:
                if i not in item_ratings or j not in item_ratings:
                    sims[i, j] = -2
                    continue
                means_prime =(means[i], means[j]) if means != None else (0, 0)
                sims[i, j] = cos_sim(item_ratings[i], item_ratings[j], num_users, 
                                     means_prime)
    
    return sims

@jit(nopython=True)
def cos_sim(item_ratings_1, item_ratings_2, num_users, means):
    a = 0
    b = 0
    c = 0
    for user in range(num_users):
        if user in item_ratings_1 and user in item_ratings_2:

            r_1 = item_ratings_1[user] - means[0]
            r_2 =  item_ratings_2[user] - means[1]
            a += (r_1)*(r_2)
            b += r_1**2
            c += r_2**2

    b = np.sqrt(b)
    c = np.sqrt(c)
    sim = a/(b*c) if b*c != 0 else -2

    return sim

def predict(sims, prefs, item, k):
    _, items = prefs.nonzero()

    # Get neighbours
    neighbs = []
    for j in items:
        sim = sims[item, j] if j >= item else sims[j, item]
        neighbs.append((sim, j))

    neighbs.sort(key=lambda x: x[0], reverse=True)
    neighbs = neighbs[:k]

    # Compute rating
    a = 0
    b = 0
    for sim, j in neighbs:
        r = prefs[0, j] - 1
        r = -1 if r == 0 else r
        a += sim*r
        b += sim

    pred = a/b if b != 0 else -2
    pred = a 
    return pred

def top_n(n, sims, user_prefs, num_items, k, iufs=None):
    _, items = user_prefs.nonzero()
    top = []
    for i in range(num_items):
        pred = predict(sims, user_prefs, i, k)
        if iufs and i in iufs:
            pred *= iufs[i]
        top.append((pred, i))

    top = [(pred, i) for pred, i in top if i not in items]
    top.sort(key=lambda x: x[0], reverse=True)
    top = top[:n]

    return top

def ensemble_top_n(n , sims_pairs, user_prefs, num_items, k):
    _, items = user_prefs.nonzero()
    top = []
    for i in range(num_items):
        pred = 0
        for sims, weight in sims_pairs:
            pred += weight*predict(sims, user_prefs, i, k)

        top.append((pred, i))

    top = [(pred, i) for pred, i in top if i not in items]
    random.shuffle(top)
    top.sort(key=lambda x: x[0], reverse=True)
    top = top[:n]

    return top

class ItemKNN:
    """Item-based neighbourhood model for item recommendation."""
    def __init__(self, k: int = 40, mean_centered: bool = False, 
                 iuf: bool = False) -> None:
        """Initialise new ItemKNN instance.
        
        Parameters:
            k - Number of neighbours
            mean_centered - True to mean center ratings.
            iuf - True to use inverse user frequency factor.
        """
        self._k = k
        self._mean_centered = mean_centered
        self._item_means = None
        self._iuf = iuf

    def fit(self, M: np.Array) -> None:
        """Train the model using utility matrix M.

        Parameters:
            M - UxI utility matrix containing user-item ratings. U is the number
                of users, I is the number of items.
        """
        self._M = M
        self._num_users, self._num_items = self._M.shape
        self._store_rating_pairs(self._M)
        if self._mean_centered:
            self._store_item_means(self._M)
        
        self._sims = compute_sims(self._item_ratings, self._num_users, 
                                  self._num_items, means=self._item_means)

    def top_n(self, user: int, n: int, prefs: np.Array = None) -> (
            list[tuple[float, int]]):
        """Return the top-n recommendations for user.
        
        Parameters:
            user - id of user.
            n - Number of recommendations to generate.
            prefs - 1xI array of user-item ratings for the given user. Default
                will try to use ratings from training.

        Returns:
            List of recommendations in the form (prediction, item_id)
        """
        try:
            prefs = self._M[[user], :] if prefs == None else prefs
        except IndexError:
            print("Could not find user. Please provide preference array.")
            return
        
        prefs = prefs.tocsr()
        top = top_n(n, self._sims, prefs, self._num_items, self._k, iufs=self._iufs)
        return top

    def _store_rating_pairs(self, M: np.Array) -> None:
        print("Storing ratings in dictionary...")
        # Type defs
        index_type = nb.types.int64
        rating_type = nb.types.float64
        dict_type = nb.types.DictType(index_type, rating_type)

        # Init mappings
        self._user_ratings = nb.typed.Dict.empty(
            key_type=index_type, 
            value_type=dict_type)
        self._item_ratings = nb.typed.Dict.empty(
            key_type=index_type, 
            value_type=dict_type)

        # Create mappings
        users, items = M.nonzero()
        num_ratings = len(users)
        for i in range(num_ratings):
            user, item = users[i], items[i]

            if user not in self._user_ratings:
                self._user_ratings[user] = nb.typed.Dict.empty(
                    key_type=index_type, value_type=rating_type)
            if item not in self._item_ratings:
                self._item_ratings[item] = nb.typed.Dict.empty(
                    key_type=index_type, value_type=rating_type)

            self._user_ratings[user][item] = M[user, item]
            self._item_ratings[item][user] = M[user, item]

        print("Done storing in dictionary.")

    def _store_item_means(self, M: np.Array) -> None:
        print("Computing item means...")
        index_type = nb.types.int64
        rating_type = nb.types.float64

        self._item_means = nb.typed.Dict.empty(key_type=index_type, value_type=rating_type)
        self._iufs = nb.typed.Dict.empty(key_type=index_type, value_type=rating_type)

        for i in range(self._num_items):
            if i % 100 == 0:
                print("Done item", i + 1, "/", self._num_items)

            if i in self._item_ratings:
                ratings = self._item_ratings[i].values()
                mean = sum(ratings)/len(self._item_ratings[i])
                self._iufs[i] = np.emath.logn(4, self._num_users
                                              /(len(self._item_ratings[i])))
            else:
                mean = 0

            self._item_means[i] = mean
        print("Done computing item means.")
            
class ContentKNN:
    """Content-based neighbourhood model for item recommendation."""
    def __init__(self, k: int = 40) -> None:
        """Initialise new ContentKNN instance.
        
        Parameters:
            k - Number of neighbours
        """
        self._k = k

    def fit(self, X: np.Array) -> None:
        """Train the model using feature array X.

        Parameters:
            X - IxF feature array. I is the number of items. F is the number of
                features.
        """
        self._sims = cosine_similarity(X)
        self._num_items = self._sims.shape[0]

    def top_n(self, user: int, n: int, prefs: np.Array = None) -> list[
        tuple[float, int]]:
        """Return the top-n recommendations for user.
        
        Parameters:
            user - id of user.
            n - Number of recommendations to generate.
            prefs - 1xI array of user-item ratings for the given user. Must be
                given.

        Returns:
            List of recommendations in the form (prediction, item_id)
        """
        if not prefs:
            print("Please give prefs array.")
            return
        prefs = prefs.tocsr()
        top = top_n(n, self._sims, prefs, self._num_items, self._k)
        return top

class EnsembleKNN:
    """Ensembler for different neighbourhood models. Takes weighted average of 
    each model to make predictions.
    """
    def __init__(self, k: int = 40):
        """Initialise new EnsembleKNN instance.
        
        Parameters:
            k - Number of neighbours
        """
        self._k = k

    def set_sims(self, sim_pairs: list[tuple[np.Array, float]]) -> None:
        """Set the similarity from each neighbourhood model.
        
        Parameters:
            sim_pairs - List of tuples of the form (similarities, weight).
                Where similarites is a IxI similarity matrix and weight is
                is the desired weight of prediction from model associated with
                similarities."""
        self._num_items = sim_pairs[0][0].shape[0]
        self._sim_pairs = sim_pairs

    def top_n(self, user: int, n: int, prefs: np.Array = None):
        """Return the top-n recommendations for user.
        
        Parameters:
            user - id of user.
            n - Number of recommendations to generate.
            prefs - 1xI array of user-item ratings for the given user. Must be
                given.

        Returns:
            List of recommendations in the form (prediction, item_id)
        """
        if not prefs:
            print("Please give prefs array.")
            return
        prefs = prefs.tocsr()
        top = ensemble_top_n(n, self._sim_pairs, prefs, self._num_items, self._k)
        return top


    