# game-rec
## RecData.py
Contains class for processing standard python dataframe into format more consumable by recommenders.

## SVD.py
Contains classes for traditional unconstrained matrix factorization and a logistic variant that uses binary cross-entropy loss. Both models use stochastic gradient descent to optimize.

## KNN.py
Contains classes for Item-Based neighbourhood collaborative filtering and content based collaborative filtering. Contains class for ensembling neighbourhood methods.

## eda.ipynb
Simple data exploration notebook.

## model.ipynb
Main notebook used for exploring svd methods.

## metadata.ipynb
Main notebook for exploring neighbourhood methods.

## Related Repositories
### game-rec-app
Video game recommendation web-app built using the KNN model can be found [here](https://github.com/MitchellJC/game-rec-app).

### Image Collection
Image collection repository can be found [here](https://github.com/MitchellJC/game-rec-scrape).
