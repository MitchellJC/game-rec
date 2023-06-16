import numpy as np
from scipy.sparse import csr_array

class RecData:
    def create_from_dataframe(self, data):
        """Create rec data from a Pandas dataframe. Columns must be in the form [item-id, rating, user_id]"""
        # Create user-item rating matrix
        self._M = data.pivot_table(index=data.columns[2], columns=data.columns[0], values=data.columns[1], aggfunc='mean')
        
        self._userid_to_index = {user_id: i for i, user_id in enumerate(self._M.index)}
        self._itemid_to_index = {app_id: i for i, app_id in enumerate(self._M.columns)}
        self._index_to_userid = {i: user_id for user_id, i in self._userid_to_index.items()}
        self._index_to_itemid = {i: app_id for app_id, i in self._itemid_to_index.items()}
        
        self._users = self._userid_to_index.keys()
        self._items = self._itemid_to_index.keys()
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
                        
    def create_anti_set(self):
        """Return all user-item pairs not in the data"""
        anti_set = []
        print(self._M.shape)
        for user in range(self._M.shape[0]):
            if user % 1000 == 0:
                print(user)
                
            for item in range(self._M.shape[1]):
                if self._M[user, item] == 0:        
                    anti_set.append((user, item))
                    
        return anti_set
            
    def get_matrix(self):
        return self._M
    
    def get_num_users(self):
        return self._num_users
    
    def get_num_items(self):
        return self._num_items
    
    def set_titles(self, titles):
        pass