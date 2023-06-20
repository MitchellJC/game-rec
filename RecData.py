import numpy as np
from scipy.sparse import csr_array, lil_array
from sklearn.preprocessing import MaxAbsScaler

class RecData:
    def create_from_dataframe(self, data):
        """Create rec data from a Pandas dataframe. Columns must be in the form [item-id, rating, user_id]"""
        # Create user-item rating matrix
        self._M = data.pivot_table(index=data.columns[2], columns=data.columns[0], values=data.columns[1], aggfunc='mean')
        
        self._userid_to_index = {user_id: i for i, user_id in enumerate(self._M.index)}
        self._itemid_to_index = {app_id: i for i, app_id in enumerate(self._M.columns)}
        self._index_to_userid = {i: user_id for user_id, i in self._userid_to_index.items()}
        self._index_to_itemid = {i: app_id for app_id, i in self._itemid_to_index.items()}
        
        self._users = [user for user in self._userid_to_index.keys()]
        self._items = [item for item in self._itemid_to_index.keys()]
        self._num_users = len(self._users)
        self._num_items = len(self._items)
        
        self._M = self._M.to_numpy()
        
        # Shift ratings up to assign 0 to missing values
        self._M += 1
        self._M = np.nan_to_num(self._M)
        
        self._M = csr_array(self._M)
                
    def leave_k_out_split(self, k=1):
        M_prime = self._M.copy()
        test = []
        for user in range(self._M.shape[0]):
            possible_indices = np.nonzero(self._M[[user], :])[1]
            left_out = np.random.choice(possible_indices, k, replace=False)
            for item in left_out:
                M_prime[user, item] = 0
                test.append((user, item, self._M[user, item]))
        
        train_data = RecData()
        train_data.__dict__.update(self.__dict__)
        train_data._M = M_prime
        
        return train_data, test
            
    def get_matrix(self):
        return self._M
    
    def get_num_users(self):
        return self._num_users
    
    def get_num_items(self):
        return self._num_items
    
    def set_titles(self, titles):
        """Maps titles to item ids using given DataFrame. Columns must be of the form [item_id, title]"""
        self._index_to_title = {self._itemid_to_index[item_id]: title for _, item_id, title in titles.itertuples() if item_id in self._itemid_to_index}
        
    def index_to_title(self, index):
        return self._index_to_title[index]
    
    def top_n(self, user, n=10):
        if self._M is None:
            raise RuntimeError("Please ensure to call fit before generating top n")
            
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
    
    def transform(self, sample):
        """"Sample should be a row vector of reviews, with NaNs for missing."""
        sample += 1
        sample = np.nan_to_num(sample)
        sample = csr_array(sample)
        return sample
        
    def search_title(self, title):
        """Finds all results for title and returns the matching title and index pairs."""
        title = title.lower()
        results = []
        for key, value in self._index_to_title.items():
            value = value.lower()
            if title in value:
                results.append((value, key))
                
        return results
    
    def create_prefs(self, prefs):
        """Create a preference array from prefs tuples in the form (index, preference) 
        where preference of 1 indicates recommend and preference of 0 indicated would not recommend."""
        prefs_vec = lil_array(np.zeros([1, self._num_items]))
        for i, pref in prefs:
            pref += 1
            prefs_vec[0, i] = pref
            
        prefs_vec = csr_array(prefs_vec)
        return prefs_vec