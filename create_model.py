import pickle
import pandas as pd
import numpy as np
import random
from RecData import RecData
from KNN import ItemKNN, EnsembleKNN


def preprocess_data(recs):
    recs = recs.sort_values(by='date')
    recs = recs.drop_duplicates(subset=['user_id', 'app_id'], keep='last')

    USED_COLS = ['app_id', 'is_recommended', 'user_id']
    recs = recs[USED_COLS]

    item_data = pd.read_csv('data/games.csv')
    titles = item_data[['app_id', 'title']]

    print("Shape:", recs.shape)
    recs.sort_values(by=['user_id', 'app_id']).head()

    return recs, titles

def create_rec_data(recs, titles):
    random.seed(42)
    np.random.seed(42)
    rec_data = RecData()
    rec_data.create_from_dataframe(recs)
    rec_data.set_titles(titles)

    return rec_data

def create_knn(rec_data):
    knn = ItemKNN(k=40, mean_centered=True)
    knn.fit(rec_data.get_matrix())

    return knn

def create_ensemble(knn):
    ens_knn = EnsembleKNN(k=40)
    ens_knn.set_sims([(knn._sims, 1)])

    return ens_knn

def save_model(rec_data, ens_knn):
    # Ensure file exists
    model_dir = "saved_models/knn/ens_knn2.pkl" 
    file = open(model_dir, 'a')
    file.close()

    # Save model
    print("Saving model...")
    with open(model_dir, 'wb') as file:
        pickle.dump([rec_data, ens_knn], file)
    print("Done saving model.")

def main():
    recs = pd.read_csv('data/full_pruned.csv')
    recs, titles = preprocess_data(recs)
    rec_data = create_rec_data(recs, titles)
    knn = create_knn(rec_data)
    ens_knn = create_ensemble(knn)

    rec_data.prep_for_item_knn()
    save_model(rec_data, ens_knn)

if __name__ == '__main__':
    main()