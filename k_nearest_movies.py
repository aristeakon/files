#make necesarry imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys, os
from contextlib import contextmanager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

dataset = pd.read_csv('data/movies/ratings.csv')

movie_titles = pd.read_csv("data/movies/movies.csv")
movie_titles.head()

df = pd.merge(dataset,movie_titles,on='movieId')

moviemat = dataset.pivot_table(index='userId',columns='movieId',values='rating')

M = moviemat.replace(np.nan, 0)
global k,metric
k=4
metric='cosine' #can be changed to 'correlation' for Pearson correlation similaries

def findksimilaritems(movieId, ratings, metric=metric, k=k):
    similarities=[]
    indices=[]    
    ratings=ratings.T
    model_knn = NearestNeighbors(metric = metric, algorithm = 'brute')
    model_knn.fit(ratings)

    distances, indices = model_knn.kneighbors(ratings.iloc[movieId-1, :].values.reshape(1, -1), n_neighbors = k+1)
    similarities = 1-distances.flatten()
    print(f"{k}. items are similar to {movieId} with title {movie_titles.loc[movie_titles['movieId'] == movieId, 'title'].unique()} and category {movie_titles.loc[movie_titles['movieId'] == movieId, 'genres'].unique()}")
    print (k)
    print (movieId)
    for i in range(0, len(indices.flatten())):
        if indices.flatten()[i]+1 == movieId:
            continue;
        else:
            print(f"{i}.:Item {indices.flatten()[i]+1} (title : {movie_titles.loc[movie_titles['movieId'] == indices.flatten()[i]+1, 'title'].unique()} and category {movie_titles.loc[movie_titles['movieId'] == indices.flatten()[i]+1, 'genres'].unique()})  with similarity of {similarities.flatten()[i]}")

    return similarities,indices

similarities,indices=findksimilaritems(2872,M)