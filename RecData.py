import numpy as np
import pandas as pd
import random
from scipy.sparse import csr_array, lil_array

class RecData:
    def create_from_dataframe(self, data):
        """Create rec data from a Pandas dataframe. Columns must be in the form 
        [item-id, rating, user_id]
        
        Parameters:
            data (Pandas DataFrame)"""
        # Create user-item rating matrix
        self._num_items = data[data.columns[0]].nunique()
        self._num_users = data[data.columns[2]].nunique()

        self._M = lil_array((self._num_users, self._num_items))
        self._userid_to_index = {}
        self._itemid_to_index = {}
        self._num_ratings = 0

        print("Creating utility matrix...")
        curr_user_index = 0
        curr_item_index = 0
        for id, item_id, rating, user_id in data.itertuples():
            self._num_ratings += 1
            rating += 1

            if user_id in self._userid_to_index:
                user_index = self._userid_to_index[user_id]
            else:
                user_index = curr_user_index
                self._userid_to_index[user_id] = user_index
                curr_user_index += 1

            if item_id in self._itemid_to_index:
                item_index = self._itemid_to_index[item_id]
            else:
                item_index = curr_item_index
                self._itemid_to_index[item_id] = item_index
                curr_item_index += 1

            self._M[user_index, item_index] = rating
        print("Done utility matrix.")

        self._index_to_userid = {i: user_id for user_id, i in self._userid_to_index.items()}
        self._index_to_itemid = {i: app_id for app_id, i in self._itemid_to_index.items()}
        
        self._users = [user for user in self._userid_to_index.keys()]
        self._items = [item for item in self._itemid_to_index.keys()]
        self._num_users = len(self._users)
        self._num_items = len(self._items)
                
    def leave_k_out_split(self, k=1):
        """Generate a leave-k-out split, creates both a validation split and 
        a test split. That is, 2*k data points will be left out.
        
        Returns:
            train_data (RecData) - train split RecData object
            val - List of tuples in the form [(user, item, rating)] for validation
            test - List of tuples in the form [(user, item, rating)] for test"""
        M_prime = self._M.copy()
        val = []
        test = []
        for user in range(self._M.shape[0]):
            if user % 1e6 == 0:
                print("Done user", user)
            
            # Val holdout
            possible_indices = self._M[[user], :].nonzero()[1]
            if len(possible_indices) > k:
                left_out = np.random.choice(possible_indices, k, replace=False)
                for item in left_out:
                    M_prime[user, item] = 0
                    val.append((user, item, self._M[user, item] - 1))
            
            # Test holdout
            if len(possible_indices) > 2*k:
                possible_indices = [index for index in possible_indices 
                                if index not in left_out]
                left_out = np.random.choice(possible_indices, k, replace=False)
                for item in left_out:
                    M_prime[user, item] = 0
                    test.append((user, item, self._M[user, item] - 1))

        train_data = RecData()
        train_data.__dict__.update(self.__dict__)
        train_data._M = M_prime
        
        return train_data, val, test
    
    def train_test_split(self, test_size=0.2):
        """Generate a train test split.
        
        Parameters:
            test_size (float) - the proportion of samples to put in test set.
            
        Returns:
            train_data (RecData) - train split RecData object
            test - List of tuples in the form [(user, item, rating)] for test"""
        M_prime = self._M.copy()
        test = []

        users, items = self._M.nonzero()
        num_samples = len(users)

        for sample in random.sample(range(num_samples), int(test_size*num_samples)):
            user = users[sample]
            item = items[sample]

            M_prime[user, item] = 0
            test.append((user, item, self._M[user, item] - 1))

        train_data = RecData()
        train_data.__dict__.update(self.__dict__)
        train_data._M = M_prime
        
        return train_data, test
    
    def generate_dataframe(self):
        """Create and return Pandas DataFrame from utility matrix."""
        app_ids = []
        user_ids = []
        ratings = []

        # Build columns
        users, items = self._M.nonzero()
        num_nonzero = len(users)
        for i in range(num_nonzero):
            user, item = users[i], items[i]

            app_id = self._index_to_itemid[item]
            user_id = self._index_to_userid[user]
            rating = self._M[user, item] - 1

            app_ids.append(app_id)
            user_ids.append(user_id)
            ratings.append(rating)

        # Create DataFrame from columns
        columns = {'app_id': app_ids, 'is_recommended': ratings, 'user_id': user_ids}
        df = pd.DataFrame(columns)
        return df
            

    def get_matrix(self):
        """Returns the user-item rating matrix."""
        return self._M
    
    def get_num_users(self):
        """Returns the total number of unique users."""
        return self._num_users
    
    def get_num_items(self):
        """Returns the total number of unique items."""
        return self._num_items
    
    def set_titles(self, titles):
        """Maps titles to item ids using given DataFrame. Columns must be of the 
        form [item_id, title]
        
        Parameters:
            titles (Pandas DataFrame) - titles to assign to items"""
        self._index_to_title = {self._itemid_to_index[item_id]: title 
                                for _, item_id, title in titles.itertuples() 
                                if item_id in self._itemid_to_index}
        
    def index_to_title(self, index):
        """Return the title of the item at the given index.
        
        Parameters:
            index (int) - The item index."""
        return self._index_to_title[index]
    
    def index_to_user_id(self, index):
        return self._index_to_userid[index]
    
    def index_to_item_id(self, index):
        return self._index_to_itemid[index]
    
    def item_id_to_index(self, i):
        return self._itemid_to_index[i]
    
    def top_n(self, user, n=10):
        """Return the true top n list for user.
        
        Parameters:
            user (int) - Index of user.
            n (int) - Number of top items to get.
            
        Returns:
            top (List) - List of top items in the form (rating, item_index)."""
            
        users, items = self._M[[user], :].nonzero()
        num_samples = len(items)
        
        users_rated = []
        for i in range(len(users)):
            users_rated.append(items[i])
            
        top = []
        for i in range(num_samples):
            item = items[i]
            
            if item not in users_rated:
                continue
                
            predicted_rating = self._M[user, item]
            
            top.append((predicted_rating, item))
            top.sort(key=lambda x: x[0], reverse=True)
            top = top[:min(n, len(top))]
        
        return top
        
    def search_title(self, title):
        """Finds all results for title and returns the matching title and 
        index pairs."""
        title_lower = title.lower()
        results = []
        for key, value in self._index_to_title.items():
            value_lower = value.lower()
            if title_lower in value_lower:
                results.append((value, key))
                
        return results
    
    def create_prefs(self, prefs):
        """Create a preference array from prefs tuples in the form (index, preference) 
        where preference of 1 indicates recommend and preference of 0 indicated 
        would not recommend."""
        prefs_vec = lil_array(np.zeros([1, self._num_items]))
        for i, pref in prefs:
            pref += 1
            prefs_vec[0, i] = pref
            
        prefs_vec = csr_array(prefs_vec)
        return prefs_vec
    
    def prep_for_item_knn(self):
        del self._M