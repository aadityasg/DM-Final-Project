import numpy as np

import pandas as pd

cols = ['userId', 'movieId', 'rating', 'timestamp']
df = pd.read_csv('ratings.csv')
del df['timestamp']


user_list = df['userId'].drop_duplicates().tolist()
user_list.sort()

movie_list = df['movieId'].drop_duplicates().tolist()
movie_list.sort()

'''Sort user and movie list. And replace the user_id or movie_id by the index of the user/movie in 
the user/movie list'''

for row in range(len(df)):
    df.set_value(row, 'userId', user_list.index(df.iloc[row]['userId']))
    df.set_value(row, 'movieId', movie_list.index(df.iloc[row]['movieId']))

cols = ['user_id', 'movie_id', 'rating']
df.to_csv('ratings_processed.csv')



