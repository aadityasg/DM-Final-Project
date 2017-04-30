import re
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from numpy import nan
from collections import Counter

def analyseUserByAge():
    userHeaders = ["userId", "age", "sex", "occupation", "zipCode"]
    userDataset = pd.read_csv("../../resources/movielens-100k-dataset/u.user", sep="|", dtype=object, header=None)
    userDataset.columns = userHeaders
    
    sns.distplot([int(x) for x in userDataset["age"]])
    sns.plt.ylabel("Density")
    sns.plt.xlabel("User Age")
    sns.plt.show()

def analyseUserByOccupation():
    headers = ["userId", "age", "sex", "occupation", "zipCode"]
    userDataset = pd.read_csv("../../resources/movielens-100k-dataset/u.user", sep="|", dtype=object, header=None)
    userDataset.columns = headers
    
    sns.countplot(x = "occupation", hue = "sex", data=userDataset)
    sns.plt.ylabel("Count")
    sns.plt.xlabel("Gender")
    sns.plt.show()
    
def analyseMovieByReleaseDate():
    headers = ["movie_id", "movie_title", "release_date", "video_release_date",
              "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
              "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    movieDataset = pd.read_csv("../../resources/movielens-100k-dataset/u.item", sep="|", dtype=object, header=None)
    movieDataset.columns = headers
    
    dates = []
    for releaseDate in movieDataset["release_date"]:
        if releaseDate is not nan:
            dates.append(datetime.strptime(releaseDate, '%d-%b-%Y').year)
    
    sns.distplot(dates)
    sns.plt.ylabel("Density")
    sns.plt.xlabel("Movie Release Date")
    sns.plt.show()

def getGenre(movie):
    availableGenres = ["unknown", "Action", "Adventure", "Animation",
              "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    
    genre = None
    
    for availGenre in availableGenres:
        if movie[availGenre] == "1":
            if genre is None:
                genre = availGenre
            else:
                genre = "Multiple"
                break
    return genre

def analyseMovieByGenre():
    headers = ["movie_id", "movie_title", "release_date", "video_release_date",
              "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
              "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    movieDataset = pd.read_csv("../../resources/movielens-100k-dataset/u.item", sep="|", dtype=object, header=None)
    movieDataset.columns = headers
    
    genres = []
    for index, row in movieDataset.iterrows():
        genre = getGenre(row)
        genres.append(genre)
    
    sns.countplot(genres)
    sns.plt.ylabel("Count")
    sns.plt.xlabel("Movie Genre")
    #sns.plt.gca().invert_yaxis()
    sns.plt.show()

def analyseRatingPattern():
    """
    ratingsHeader = ["user_id", "movie_id", "rating", "timestamp"]
    ratingsDataset = pd.read_csv("../../resources/movielens-100k-dataset/u.data", sep="\t", dtype=object, header=None)
    ratingsDataset.columns = ratingsHeader
    
    moviesHeaders = ["movie_id", "movie_title", "release_date", "video_release_date",
              "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
              "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    movieDataset = pd.read_csv("../../resources/movielens-100k-dataset/u.item", sep="|", dtype=object, header=None)
    movieDataset.columns = moviesHeaders
    
    
    userHeaders = ["userId", "age", "sex", "occupation", "zipCode"]
    userDataset = pd.read_csv("../../resources/movielens-100k-dataset/u.user", sep="|", dtype=object, header=None)
    userDataset.columns = userHeaders
    
    data = {"Gender": [], "Rating":[], "Genre": [], "Occupation":[]} 
    availableGenres = ["unknown", "Action", "Adventure", "Animation",
              "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    
    parentDf = None
    counter = 0
    for idx, rating in ratingsDataset.iterrows():
        print idx
        user = userDataset.loc[userDataset['userId'] == rating["user_id"]]
        movie = movieDataset.loc[movieDataset['movie_id'] == rating["movie_id"]]
        found = re.search(".*([MF])", str(user["sex"]))
        gender = found.group(1)
        found = re.search(".* ([a-z]+)", str(user["occupation"]))
        occupation = found.group(1)
        for availGenre in availableGenres:
            if int(movie[availGenre]) == 1:
                data["Rating"].append(float(rating["rating"]))
                data["Gender"].append(gender)
                data["Genre"].append(availGenre)
                data["Occupation"].append(occupation)
                
                counter += 1
                if counter % 10000 == 0:
                    df = pd.DataFrame.from_dict(data)
                    if parentDf is None:
                        parentDf = df
                    else:
                        parentDf = pd.concat([parentDf, df])
                    print parentDf
                    data = {"Gender": [], "Rating":[], "Genre": [], "Occupation":[]}
    
    df = pd.DataFrame.from_dict(data)
    parentDf = pd.concat([parentDf, df])
    parentDf.to_csv("../../resources/movielens-100k-dataset/GenderRatingGenreOccupation.csv", sep=',', index=False, encoding='utf-8')
    """
    parentDf = pd.read_csv("../../resources/movielens-100k-dataset/GenderRatingGenreOccupation.csv", dtype=object)
    parentDf.Rating = parentDf.Rating.astype(np.float)
    
    sns.barplot(x="Genre", y="Rating", hue="Gender", data=parentDf)
    sns.plt.show()

def analyseRatingPattern2():
    df = pd.read_csv("../../resources/movielens-100k-dataset/GenderRatingGenreOccupation.csv", dtype=object)
    data = {}
    
    occupation_list = ["administrator", "artist", "doctor",
                       "educator", "engineer", "entertainment", "executive",
                       "healthcare", "homemaker", "lawyer", "librarian",
                       "marketing", "none", "other", "programmer",
                       "retired", "salesman", "scientist", "student",
                       "technician", "writer"]
    
    genres_list = ["unknown", "Action", "Adventure", "Animation",
                   "Childrens", "Comedy", "Crime", "Documentary", 
                   "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", 
                   "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
    
    """data = {"Gender": [], "Rating":[], "Genre": [], "Occupation":[]}""" 
    for _, row in df.iterrows():
        key = row["Occupation"] + "<=>" + row["Genre"]
        if key not in data.keys():
            data[key] = list()
            data[key].append(0)
            data[key].append(0)
            
        data[key][0] += float(row["Rating"])
        data[key][1] += 1
        
    print "Here!!!"
    print len(occupation_list)
    print len(genres_list)
    
    dt = np.zeros(shape=(len(occupation_list), len(genres_list)))
    for idx1, occupation in zip(range(len(occupation_list)), occupation_list):
        for idx2, genre in zip(range(len(genres_list)), genres_list):
            key = occupation + "<=>" + genre
            if key in data.keys():
                dt[idx1][idx2] = data[key][0] / float(data[key][1])
            else:
                dt[idx1][idx2] = 0
    
    fig, ax = sns.plt.subplots()
    heatmap = ax.pcolor(dt, cmap='RdBu_r')
    
    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(dt.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(dt.shape[1])+0.5, minor=False)
    
    # want a more natural, table-like display
    #ax.invert_yaxis()
    #ax.xaxis.tick_top()
    
    ax.set_xticklabels(genres_list, minor=False)
    ax.set_yticklabels(occupation_list, minor=False)
    sns.plt.show()

#df = pd.read_csv("../../resources/movielens-100k-dataset/GenreGenderRating-Final.csv", sep=',', dtype=object, header=None)
analyseRatingPattern2()
analyseRatingPattern()
analyseMovieByGenre()
analyseUserByOccupation()
analyseUserDataset()
analyseMovieByReleaseDate()