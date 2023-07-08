from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse import vstack, lil_array, csr_array
from collections import defaultdict
import time
import random
import math
import numpy as np
from queue import SimpleQueue
from sklearn.metrics.pairwise import cosine_similarity

import numba as nb
from numba import jit, njit, float64, uint32
from numba.typed import List, Dict

from RecData import RecData

class Metrics:
    def rmse(self, predictions):
        return math.sqrt(sum((prediction - true_rating)**2 for _, _, prediction, 
                             true_rating in predictions)/len(predictions))

class SVDBase():
    """Base class for funk svd"""
    def __init__(self, num_users, num_items, num_ratings, k=10, learning_rate=0.01,
                  C=0.02):
        """Initialise new SVDBase.
        
        Parameters:
            num_user (int) - Number of users
            num_items (int) - Number of items
            num_ratings (int) - The number of different possible ratings
            k (int) - The number of latent factors
            learning_rate (float) - The learning rate
            C (float) - Regularization parameter
            partial"""
        self._num_users = num_users
        self._num_items = num_items
        self._num_ratings = num_ratings
        
        self._k = k
        self._learning_rate = learning_rate
        self._C = C
        self._lrate_C = self._learning_rate*self._C
        
        self._user_features = np.array(np.random.normal(size=(self._num_users, self._k), 
                                               scale=0.01), dtype=np.float64)
        self._item_features = np.array(np.random.normal(size=(self._num_items, self._k), 
                                               scale=0.01), dtype=np.float64)
        self._user_biases = np.array(np.random.normal(size=(self._num_users, 1), 
                                               scale=0.01), dtype=np.float64)
        self._item_biases = np.array(np.random.normal(size=(self._num_items, 1), 
                                               scale=0.01), dtype=np.float64)
        
        self._M = None
        self._num_samples = None
        self._train_errors = None
        self._val_errors = None
        self._epoch = -1

    def fit(self, M, epochs, validation_set=None, tol=1e-15, early_stop=True):
        """Fit the model with the given utility matrix M (csr array).
        
        Parameters:
            M (scipy csr_array) - UxI array where U is the number of users
                and I is the number of items
            epochs (int) - The number of epochs
            validation_set (list) - List of validation pairs
            tol (float) - Early stopping tolerance
            early_stop (bool) - True to enable automatic early stopping"""
        self._M = M 
        
        self._validation_set = validation_set
        self._tol = tol
        self._train_errors = []
        if validation_set:
            self._val_errors = []
        
        # Retrieve sample locations
        self._users, self._items = self._M.nonzero()
        self._num_samples = len(self._users)
        self._mask = (self._M != 0)

        self._cache_users_rated()

        self._run_epochs(self._users, self._items, epochs, early_stop=early_stop)

    def continue_fit(self, epochs, early_stop=True):
        """Continue training for extra epochs"""          
        self._run_epochs(self._users, self._items, epochs, early_stop=early_stop)

    def partial_fit(self, new_sample, epochs, batch_size=0, compute_err=False):
        """"Faciliates online training. Add new user vector new_sample into the 
        model and fit with warm start.
        
        Parameters:
            new_sample (csr_array) - 1xI arrary where I is the number of items
            epochs (int) - The number of epochs
            batch_size (int) - The number of users to mix-in for training
            compute_err (bool) - True to enable training error computation"""
        self._num_users += 1
        self._M = csr_array(vstack([lil_array(self._M), new_sample]))
        total_users, total_items = self._M.nonzero()
        self._users, self._items = total_users, total_items
        self._num_samples = len(total_users)
        
        self._mask = (self._M != 0)
        
        self._user_features = np.concatenate(
            [self._user_features, np.random.normal(size=(1, self._k), scale=0.01)], axis=0)
                                                                               
        indices_of_new = [new_i for new_i in range(self._num_users - 1, len(total_users))]

        self._cache_users_rated()
                                              
        for epoch in range(epochs):
            start_time = time.time()
            # Choose a smaller subset of total samples already fitted
            # fitted_subset = random.sample(range(self._num_samples), k=batch_size)    
            
            # Ensure that new indices are always used
            possible_indices =  indices_of_new # + fitted_subset
            
            # Perform update for each sample
            for i in random.sample(possible_indices , k=len(possible_indices)):
                (self._user_features, self._item_features,
                 self._user_biases, self._item_biases) = (
                    update_fast(i, total_users[i], total_items[i], self._M.data, 
                                self._user_features,
                                self._item_features,
                                self._user_biases,
                                self._item_biases,
                                self._learning_rate,
                                self._lrate_C)
                ) 
            
            print("Epoch", epoch, end="/")
            if compute_err:
                self._compute_error()
            print("Time:", round(time.time() - start_time, 2), "seconds")

    def pop_user(self):
        """Remove the last added user from the model."""
        self._num_users -= 1
        self._M = self._M[:-1, :]        
        self._user_features = self._user_features[:-1, :]

    def top_n(self, user, n=10):
        """Return the top n recommendations for given user.
        
        Parameters:
            user (int) - The index of the user
            n (int) - The number of recommendations to give
            
        Preconditions:
            n > 0"""        
        top = []
        for item in range(self._num_items):
            # Do not add items for which rating already exists
            if user in self._users_rated and item in self._users_rated[user]:
                continue
                
            predicted_rating = self.predict(user, item)
            
            top.append((predicted_rating, item))
            top.sort(key=lambda x: x[0], reverse=True)
            top = top[:min(n, len(top))]
        
        return top
    
    def predict_pairs(self, pairs):
        """Returns a list of predictions of the form (user, item, prediction) 
        for each (user, item) pair in pairs.
        
        Parameters:
            pairs (list) - List of (user, item) tuples.
            
        Returns:
            List of (user, item, prediction) tuples."""
        predictions = []
        for user, item in pairs:
            prediction = self.predict(user, item)
            predictions.append((user, item, prediction))
        
        return predictions

    def get_train_errors(self):
        """Return the training errors stored while training. Returns none if 
        model has not been fit."""
        return self._train_errors
    
    def get_val_errors(self):
        """Return the validation errors stored while training. Returns none if 
        model has not been fit."""
        return self._val_errors
    
    def prep_for_item_knn(self):
        del self._user_features
        del self._user_biases
        del self._item_biases
        del self._M
        del self._mask

    def _cache_users_rated(self):
        self._users_rated = {}
        for sample_num in range(self._num_samples):
            user = self._users[sample_num]
            item = self._items[sample_num]
            if user not in self._users_rated:
                self._users_rated[user] = []
            self._users_rated[user].append(item)
            
    def _run_epochs(self, users, items, epochs, early_stop=False):
        self._M = csr_array(self._M)
        for epoch in range(epochs):
            self._epoch += 1
            start_time = time.time()
            
            # For all samples in random order update each parameter
            for i in random.sample(range(self._num_samples), k=self._num_samples):
                (self._user_features, self._item_features, 
                self._user_biases, self._item_biases) = (
                    update_fast(i, users[i], items[i], self._M.data, 
                                self._user_features,
                                self._item_features,
                                self._user_biases,
                                self._item_biases,
                                self._learning_rate,
                                self._lrate_C)
                ) 
            
            # Display training information
            print("Epoch", epoch, end="/")
            loss = self._compute_error()
            print("Training error:", loss, end="/")

            if self._validation_set:
                val_loss = self._compute_val_error()
                print("Validation error:", val_loss, end="/")
                
            print("Time:", round(time.time() - start_time, 2), "seconds")
            
            # Convergence criterion
            if (self._validation_set 
                and early_stop 
                and epoch > 1 
                and self._val_errors[-2] - self._val_errors[-1] < self._tol
            ):
                    print("Small change in validation error. Terminating training.")
                    return
            
            
class RatingSVD(SVDBase):
    """SVD for collaborative filtering, uses explicit ratings."""
    def fit(self, M, epochs, validation_set=None, tol=1e-15, early_stop=True):
        self._M = M 
        self._M = self._M.tocsr()
        
        self._validation_set = validation_set
        self._tol = tol
        self._train_errors = np.zeros([epochs])
        if validation_set:
            self._val_errors = np.zeros([epochs])
        
        # Retrieve sample locations
        self._users, self._items = self._M.nonzero()
        self._num_samples = len(self._users)
        self._mask = (self._M != 0)

        self._cache_users_rated()
        self._cache_user_item_weights()

        self._run_epochs(self._users, self._items, epochs, early_stop=early_stop)

    def predict(self, user, item):
        return predict_fast_rating(user, item, self._user_features, 
                                  self._item_features, self._user_biases, 
                                  self._item_biases)
    
# Fast Numba Methods
################################################################################
@jit(nopython=True)
def predict_fast_rating(user, item, user_features, item_features, user_biases, item_biases):
    return (np.dot(user_features[user, :], item_features[item, :]) 
            + user_biases[user, 0] + item_biases[item, 0])

################################################################################

class LogisticSVD(SVDBase):
    def fit(self, M, epochs, validation_set=None, tol=1e-15, early_stop=True):
        self._M = M 
        self._M = self._M.tocsr()
        
        self._validation_set = validation_set
        self._tol = tol
        self._train_errors = np.zeros([epochs])
        if validation_set:
            self._val_errors = np.zeros([epochs])
        
        # Retrieve sample locations
        self._users, self._items = self._M.nonzero()
        self._num_samples = len(self._users)
        self._mask = (self._M != 0)

        self._cache_users_rated()
        self._cache_user_item_weights()

        self._run_epochs(self._users, self._items, epochs, early_stop=early_stop)

    def continue_fit(self, epochs, early_stop=True):
        """Continue training for extra epochs"""      
        new_train_errors = np.zeros([self._train_errors.shape[0] + epochs])
        new_val_errors = np.zeros([self._train_errors.shape[0] + epochs])   

        new_train_errors[:self._train_errors.shape[0]] = self._train_errors
        new_val_errors[:self._val_errors.shape[0]] = self._val_errors

        self._train_errors = new_train_errors
        self._val_errors = new_val_errors
        self._run_epochs(self._users, self._items, epochs, early_stop=early_stop)

    def predict(self, user, item):
        return predict_fast(user, item, self._user_features, 
                                  self._item_features, self._user_biases, self._item_biases)
    
    def compute_sims(self):
        start_t = time.time()
        q = self._item_features
        self._sims = lil_array((q.shape[0], q.shape[0]))
        print("Computing similarities...")
        for i in range(q.shape[0]):
            if i % 200 == 0:
                print("Upto row", i)
            for j in range(i, q.shape[0]):
                self._sims[i, j] = cosine_similarity(q[[i], :], q[[j], :])
        print("Done computing similarities in", time.time() - start_t, "seconds")
    
    def items_knn(self, subjects, n=10):
        alpha = 0.1
        top = []
        disliked = [(i, pref) for i, pref in subjects if pref == 0]
        # Get candidates
        for i, pref in subjects:
            if pref == 0:
                continue
            
            for j in range(self._sims.shape[0]):
                if i == j:
                    continue
                elif j < i:
                    sim = self._sims[j, i]
                elif j > i:
                    sim = self._sims[i, j]

                # Get min dissimilarity
                dissims = [1]
                for k, pref in disliked:
                    if j == k:
                        continue
                    elif j < k:
                        dissim = 1 - self._sims[j, k]*alpha
                    elif j > k:
                        dissim = 1 - self._sims[k, j]*alpha

                    dissims.append(dissim)

                sim *= min(dissims)

                top.append((sim, j))
                top.sort(reverse=True)
                top = top[:10*n]

        # Filter already seen
        seen = [id for id, pref, in subjects]
        top = [cand for cand in top if cand[1] not in seen]

        # Filter duplicates
        new_top = []
        seen = []
        for score, id in top:
            if id in seen:
                continue

            seen.append(id)
            new_top.append((score, id))
        top = new_top

        top = top[:n]
        return top

    # def top_n(self, user, n=10):
    #     """Return the top n recommendations for given user.
        
    #     Parameters:
    #         user (int) - The index of the user
    #         n (int) - The number of recommendations to give
            
    #     Preconditions:
    #         n > 0"""        
    #     prefs = self._M[[user], :]
    #     users, items = prefs.nonzero()
    #     num_prefs = len(users)
    #     prefs = [(items[i], prefs[0, items[i]] - 1) for i in range(num_prefs)]
    #     top = self.items_knn(prefs, n=n)
    
    #     return top  
    
    def top_n(self, user, n=10):
        """Return the top n recommendations for given user.
        
        Parameters:
            user (int) - The index of the user
            n (int) - The number of recommendations to give
            
        Preconditions:
            n > 0"""        
        top = []
        for item in range(self._num_items):
            # Do not add items for which rating already exists
            if user in self._users_rated and item in self._users_rated[user]:
                continue
                
            predicted_rating = self.predict(user, item)
            
            top.append((predicted_rating, item))
            top.sort(key=lambda x: x[0], reverse=True)
            top = top[:min(n, len(top))]
        
        return top

    def compute_recall(self, test, k=20):
        tops = {}
        hits = 0
        num_rel = 0

        # Generate top lists
        i = 0
        for user, item, rating in test:
            if i % 10000 == 0:
                print("Generating top-n for user", i)

            if user not in tops:
                tops[user] = [item for _, item in self.top_n(user, n=k)]
            
            if rating == 1:
                num_rel += 1

            i += 1

        # See how many hits we got
        for user, item, rating in test:
            if item in tops[user]:
                hits += 1

        recall = hits / num_rel
        self._recall = recall
        print(recall)

    def _cache_users_rated(self):
        self._users_rated = {}
        for sample_num in range(self._num_samples):
            user = self._users[sample_num]
            item = self._items[sample_num]
            if user not in self._users_rated:
                self._users_rated[user] = []
            self._users_rated[user].append(item)

    def _cache_user_item_weights(self):
        user_freqs = np.zeros([self._num_users, 1])
        item_freqs = np.zeros([self._num_items, 1])
        users, items = self._M.nonzero()
        num_samples = len(users)

        for i in range(num_samples):
            user, item = users[i], items[i]

            user_freqs[user, 0] += 1
            item_freqs[item, 0] += 1

        self._user_props = ( user_freqs - np.min(user_freqs) )/ (np.max(user_freqs) - np.min(user_freqs))
        self._item_props = ( item_freqs - np.min(item_freqs) )/ (np.max(item_freqs) - np.min(item_freqs))
    
    

    def _compute_error(self):
        self._M = self._M.tocsr()
        return compute_error_fast(self._M.data, self._M.indices, self._M.indptr, 
                           self._num_samples, self._train_errors, self._epoch,
                           self._user_features, self._item_features, self._user_biases, self._item_biases)
        
    def _compute_val_error(self):
        return compute_val_error_fast(self._val_errors, List(self._validation_set), 
                                      self._epoch, self._user_features, 
                                      self._item_features, self._user_biases, self._item_biases)

# Fast Numba Methods
################################################################################
@jit(nopython=True)
def update_fast(i, user, item, values, user_features, item_features, user_biases, 
                item_biases, learning_rate, lrate_C, do_items=True):
    # Pre-cache computations
    true = values[i] - 1
    pred = predict_fast(user, item, user_features, 
                        item_features, user_biases, item_biases)
    a = np.exp( -( np.dot(user_features[user, :], item_features[item, :]) 
                  + user_biases[user, 0] + item_biases[item, 0] )
    )
    ab = a*pred
    coeff = learning_rate*( 
        ( -(1 - true)*ab*pred )/(1 - pred) + true*ab 
        )
    
    # Compute user features update
    new_user_features = (
        user_features[user, :] + item_features[item, :]*coeff
        -lrate_C*user_features[user, :]
    )

    new_user_biases = (
        user_biases[user, 0] + coeff # - lrate_C*user_biases[user, 0]
    )
    
    if do_items:
        # Compute item features update
        new_item_features = (
            item_features[item, :] + user_features[user, :]*coeff
            -lrate_C*item_features[item, :]
        )
        new_item_biases = (
            item_biases[item, 0] + coeff # - lrate_C*item_biases[item, 0]
        )

        item_features[item, :] = new_item_features
        item_biases[item, 0] = new_item_biases

    user_features[user, :] = new_user_features
    user_biases[user, 0] = new_user_biases

    return user_features, item_features, user_biases, item_biases

@jit(nopython=True)
def compute_error_fast(values, indices, indptr, num_samples, train_errors, 
                       epoch, user_features, item_features, user_biases, item_biases):
    loss = 0

    num_vals = 0
    next_row_index = 0
    for i in range(len(values)):
        value = values[i]
        column = indices[i]

        num_vals += 1
        if indptr[next_row_index] < num_vals:
            next_row_index += 1

        true = value - 1
        pred = predict_fast(next_row_index - 1, column, user_features, item_features, user_biases, item_biases)

        loss += true*np.log(pred) + (1 - true)*np.log(1 - pred)

    loss *= -(1/num_samples)
    train_errors[epoch] = loss

    return loss

@jit(nopython=True)
def compute_val_error_fast(val_errors, validation_set,
                       epoch, user_features, item_features, user_biases, item_biases):
   # Predict rating for all pairs in validation
    predictions = predict_pairs_fast([(user, item) 
                                for user, item, _ in validation_set], 
                                user_features, item_features, user_biases, item_biases)
    
    # Add true ratings into tuples
    predictions = [prediction + (validation_set[i][2],) 
                    for i, prediction in enumerate(predictions)]
    
    val_error = 0
    for user, item, pred, true in predictions:
        
        val_error += true*np.log(pred) + (1 - true)*np.log(1 - pred)

    val_error *= -(1/len(predictions))
    val_errors[epoch] = val_error

    return val_error

@jit(nopython=True)
def sigmoid_fast(x):
    # if x.shape == (1, 1) and np.isnan(x):
    #     print("x is nan!!!!!!!!")
    #     raise ValueError()
    return 1/(1 + np.exp(-x))

@jit(nopython=True)
def predict_fast(user, item, user_features, item_features, user_biases, item_biases):
    sig = sigmoid_fast(
            np.dot(user_features[user, :], item_features[item, :]) + user_biases[user, 0] + item_biases[item, 0]
            )
    sig = np.minimum(sig, 0.9999)
    sig = np.maximum(sig, 0.00001)
    if np.isnan(sig):
        sig = 0.9999
    return sig

@jit(nopython=True)
def predict_pairs_fast(pairs, user_features, item_features, user_biases, item_biases):
        """Returns a list of predictions of the form (user, item, prediction) 
        for each (user, item) pair in pairs.
        
        Parameters:
            pairs (list) - List of (user, item) tuples.
            
        Returns:
            List of (user, item, prediction) tuples."""
        predictions = []
        for user, item in pairs:
            prediction = predict_fast(user, item, user_features, item_features, user_biases, item_biases)
            predictions.append((user, item, prediction))
        
        return predictions

