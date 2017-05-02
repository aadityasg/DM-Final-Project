import numpy as np
from numpy.linalg import inv
import pandas as pd
from sklearn.metrics import mean_squared_error


import math as math

def get_mse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()

    actual = actual[actual.nonzero()].flatten()
    return mean_squared_error(pred, actual)

''' Number of latent factors'''

cols = ['user_id', 'movie_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=cols)
del df['timestamp']


user_list = df['user_id'].drop_duplicates().tolist()
user_list.sort()

movie_list = df['movie_id'].drop_duplicates().tolist()
movie_list.sort()


num_users = len(user_list)
num_movies = len(movie_list)


def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    for user in xrange(ratings.shape[0]):
        count = np.count_nonzero(ratings[user,: ])
        count = int(count * 0.2)
        # Take 20% of each row's non zero elements as test elements
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],
                                        size=count,
                                        replace=False)

        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]

    return train, test


'''Creating a rating matrix'''

ratings = np.zeros((num_users, num_movies))


for row in range(len(df)):

    ratings[int(df.iloc[row]['user_id']-1), int(df.iloc[row]['movie_id'])-1] = df.iloc[row]['rating']


def predict(u, i):
    global user_matrix
    global movie_matrix

    return user_matrix[u, :].dot(movie_matrix[i, :].T)

##The Latent Vector for the User matrix


def als_user(movie_mat, user_mat, ratings):

    global reg_user
    global num_users

    YTY = movie_mat.T.dot(movie_mat)

    const_user = np.eye(YTY.shape[0]) * reg_user

    for u in range(num_users):

        user_mat[u, :] = ratings[u, :].dot(movie_mat).dot(inv(YTY + const_user))

    return user_mat


def als_movie(movie_mat, user_mat, ratings):

    global reg_movies

    global num_movies

    XTX = user_mat.T.dot(user_mat)

    const_movies = np.eye(XTX.shape[0]) * reg_movies

    for i in range(num_movies):

        movie_mat[i, :] = ratings[:, i].T.dot(user_mat).dot(inv(XTX + const_movies))

    return movie_mat


itern_list = [50]
k_list = [10,20,30,40,50]
lambda_list = np.logspace(-3, 2.0, num=10)



w = 0.005
threshold = 0.55
df1 = pd.read_csv('similarities-final.csv')
d = df1.values

def sim(movie_id_1, movie_id_2):

    global d
  
    if math.isnan(d[movie_id_1-1][movie_id_2 - 1]):
        return 0
    return d[movie_id_1-1][movie_id_2 - 1]



t_list = [0.5, 0.55]
min_val = 300
for threshold in t_list:
    print "Threshold = "+str(threshold)
    data = []
    for lambda_ in lambda_list:

        reg_user = lambda_
        reg_movies = lambda_
        train, test = train_test_split(ratings=ratings)
        print " --------- lambda = " + str(lambda_) + "-------------"
        row = []
        for k in k_list:
            print "k = " + str(k)
            for iter_cur in itern_list:

                user_matrix = np.random.random((num_users, k))
                movie_matrix = np.random.random((num_movies, k))
                for j in range(iter_cur):

                    movie_matrix = als_movie(movie_mat=movie_matrix, user_mat=user_matrix, ratings=train)

                    user_matrix = als_user(movie_mat=movie_matrix, user_mat=user_matrix, ratings=train)


                predictions = np.zeros((num_users, num_movies))
                for u in xrange(num_users):
                    user_ratings_copy = []
                    user_ratings_copy = list(ratings[u, :])
                    
                    max_rating = max(user_ratings_copy)
                    representatives = [q for q, x in enumerate(user_ratings_copy) if x == max_rating]
                    for i in xrange(num_movies):
                        predictions[u, i] = predict(u, i)
                        for cur_representative_movie in representatives:
                            
                            similarity = sim(cur_representative_movie + 1, i + 1)
                            if similarity > threshold and (cur_representative_movie != i):
                                predictions[u, i] += (w * max_rating * similarity)

                pred_max, pred_min = predictions.max(), predictions.min()
                predictions = (predictions - pred_min) / (pred_max - pred_min)


                for u in xrange(num_users):
                    for i in xrange(num_movies):
                        predictions[u, i] = round(predictions[u, i] * 5, 0)
                test_ms = get_mse(predictions, test)
                if min_val > test_ms:
                    min_val = test_ms

                row.append(test_ms)
                print "Training MSE: "+str(get_mse(predictions, train)) + " Test MSE: " + str(test_ms)

        data.append(row)


    print data

print min_val

