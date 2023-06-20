from scipy.sparse.linalg import norm as sparse_norm
from scipy.sparse import vstack, lil_array
from collections import defaultdict
import time
import random
import math
import numpy as np

class Metrics:
    def rmse(self, predictions):
        return math.sqrt(sum((prediction - true_rating)**2 for _, _, prediction, 
                             true_rating in predictions)/len(predictions))

class SVDPredictor:
    """SVD for collaborative filtering"""
    def __init__(self, num_users, num_items, k=100, learning_rate=0.01, epochs=5,
                  C=0.02, partial_batch_size=int(1e5)):
        self._num_users = num_users
        self._num_items = num_items
        
        self._k = k
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._C = C
        self._partial_batch_size = partial_batch_size
        
        self._user_features = np.random.normal(size=(self._num_users, self._k), 
                                               scale=0.01)
        self._item_features = np.random.normal(size=(self._num_items, self._k), 
                                               scale=0.01)
        self._implicit_features = np.random.normal(size=(self._num_items, self._k), 
                                                   scale=0.01)
        self._user_biases = np.zeros([self._num_users, 1])
        self._item_biases = np.zeros([self._num_items, 1])
        
        self._M = None
        self._num_samples = None
        self._train_errors = None
        self._val_errors = None
    
    def fit(self, M, validation_set=None):
        """Fit the model with the given user-item matrix (csr array)."""
        self._M = M 
        self._train_errors = []
        if validation_set:
            self._val_errors = []
            
        # Retrieve sample locations
        users, items = M.nonzero()
        self._num_samples = len(users)
        self._mask = (M != 0)
        
        self._mu = self._M.sum() / self._num_samples
        
        for epoch in range(self._epochs):
            start_time = time.time()
            
            # For all samples in random order update each parameter
            for i in random.sample(range(self._num_samples), k=self._num_samples):
                self._update_features(i, users, items)                
            
            # Display training information
            print("Epoch", epoch, end="/")
            self._show_error()
            
            if validation_set:
                # Predict rating for all pairs in validation
                predictions = self.predict([(user, item) 
                                            for user, item, _ in validation_set])
                
                # Add true ratings into tuples
                predictions = [prediction + (validation_set[i][2],) 
                               for i, prediction in enumerate(predictions)]
                
                metrics = Metrics()
                val_error = metrics.rmse(predictions)
                self._val_errors.append(val_error)
                print("Validation error:", val_error, end="/")
                
            print("Time:", round(time.time() - start_time, 2), "seconds")
            
            # Convergence criterion
            if validation_set:
                if (len(self._val_errors) > 1 
                    and self._val_errors[-2] - self._val_errors[-1] < 1e-14
                    ):
                    print("Very small change in validation error. Terminating training.")
                    return
                    
            
    def partial_fit(self, new_sample):
        """"Faciliates online training. Add new user vector new_sample into the 
        model and fit with warm start."""
        
        users, items = self._M.nonzero()
        self._M = vstack([self._M, new_sample])
        total_users, total_items = self._M.nonzero()
        
        self._mask = (self._M != 0)
        
        num_samples = len(users)
        new_sample_index = self._num_users
        self._num_users += 1
        
        self._user_features = np.concatenate(
            [self._user_features, np.random.normal(size=(1, self._k), scale=0.01)], axis=0)
        self._user_biases = np.concatenate([self._user_biases, np.zeros([1, 1])], axis=0)
                                                                               
        indices_of_new = [new_i for new_i in range(len(users), len(total_users))]
                                              
        for epoch in range(self._epochs):
            start_time = time.time()
            # Choose a smaller subset of total samples already fitted
            fitted_subset = random.sample(range(num_samples), k=self._partial_batch_size)    
            
            # Ensure that new indices are always used
            possible_indices = fitted_subset + indices_of_new
            
            # Perform update for each sample
            for i in random.sample(possible_indices , k=len(possible_indices)):
                self._update_features(i, total_users, total_items, do_items=False)
            
            print("Epoch", epoch, end="/")
            self._show_error()
            print("Time:", round(time.time() - start_time, 2), "seconds")
            
        
    def top_n(self, user, n=10):
        """Return the top n recommendations for given user.
        
        Parameters:
            user (int) - The index of the user
            n (int) - The number of recommendations to give
            
        Preconditions:
            n > 0"""
        if self._M is None:
            raise RuntimeError("Please ensure to call fit before generating top n")
        users, items = self._M[[user], :].nonzero()
        
        users_rated = []
        for i in range(len(users)):
            users_rated.append(items[i])
        
        top = []
        for item in range(self._num_items):
            # Do not add items for which rating already exists
            if item in users_rated:
                continue
                
            predicted_rating = (
                self._mu + self._user_biases[user, 0] 
                + self._item_biases[item, 0] 
                + self._user_features[user, :] @ np.transpose(self._item_features)[:, item])
            
            top.append((predicted_rating, item))
            top.sort(key=lambda x: x[0], reverse=True)
            top = top[:min(n, len(top))]
        
        return top
        
    def predict(self, pairs):
        """Returns a list of predictions of the form (user, item, prediction) 
        for each (user, item) pair in pairs.
        
        Parameters:
            pairs (list) - List of (user, item) tuples.
            
        Returns:
            List of (user, item, prediction) tuples."""
        predictions = []
        for user, item in pairs:
            prediction = (self._mu + self._user_biases[user, 0] 
                          + self._item_biases[item, 0] 
                          + self._user_features[user, :] @ np.transpose(self._item_features)[:, item])
            prediction = prediction
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
    
    def _update_features(self, i, users, items, do_items=True):
        user = users[i]
        item = items[i]                  

        diff = (
            self._M[user, item] 
            - (self._mu + self._user_biases[user, 0] + self._item_biases[item, 0] 
           + self._user_features[user, :] @ np.transpose(self._item_features)[:, item])
           )
        
        # Compute user bias update
        self._user_biases[user, 0] += self._learning_rate*(
            diff - self._C*self._user_biases[user, 0])
        

        # Compute user features update
        new_user_features = (self._user_features[user, :] + self._learning_rate*(
            diff*self._item_features[item, :] 
            - self._C*self._item_features[item, :]))
        
        if do_items:
            # Compute item features update
            self._item_features[item, :] += self._learning_rate*(
                diff*self._user_features[user, :] 
                - self._C*self._user_features[user, :])
            
            # Compute item bias update
            self._item_biases[item, 0] += self._learning_rate*(
                diff - self._C*self._item_biases[item, 0])

        self._user_features[user, :] = new_user_features
        
    def _show_error(self):
        big_diff = (
            self._M - (self._mu + np.repeat(self._user_biases, self._M.shape[1], axis=1) 
                       + np.repeat(np.transpose(self._item_biases), self._M.shape[0], axis=0) 
                       + self._user_features @ np.transpose(self._item_features)))
        
        # Mask to ignore error from missing reviews
        big_diff = self._mask.multiply(big_diff)
        error = sparse_norm(big_diff) / np.sqrt(self._num_samples)
        self._train_errors.append(error)
        print("Training error:", error, end="/")