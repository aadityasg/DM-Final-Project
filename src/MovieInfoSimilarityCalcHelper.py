import numpy as np
import math
import pandas as pd
from PlotSimilarityCalcHelper import calculatePlotSimilarity
from multiprocessing.pool import ThreadPool
import logging

logging.basicConfig(filename='similarityrun.log',level=logging.DEBUG)

cache = {}


def computeMovieSimilarity(movie1, movie2):
    #print movie1["title"] + " vs " + movie2["title"]
    
    headers = ["movie_id", "movie_title", "release_date", "video_release_date",
              "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
              "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    
    genresList = ["unknown", "Action", "Adventure", "Animation",
                  "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
                  "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
                  "Thriller", "War", "Western"]
    
    if movie1["imdbRating"] == "?":
        movie1Rating = 5
    else:
        movie1Rating = float(movie1["imdbRating"])
    if movie2["imdbRating"] == "?":
        movie2Rating = 5
    else:
        movie2Rating = float(movie2["imdbRating"])
    
    actors1 = [actor.strip() for actor in str(movie1["Actors"]).split(",")]
    actors2 = [actor.strip() for actor in str(movie2["Actors"]).split(",")]
    actors = set(actors1)
    actors |= set(actors2)
    commonActors = [actor for actor in actors1 if actor in actors2]
    actorsSimilarityScore = len(commonActors) / float(len(actors)) 
    
    
    genres1 = [genre for genre in genresList if movie1[genre] == "1"]
    genres2 = [genre for genre in genresList if movie2[genre] == "1"]
   
    genres = set(genres1)
    genres |= set(genres2)
    commonGenres = [genre for genre in genres1 if genre in genres2]
    genresSimilarityScore = len(commonGenres) / float(len(genres)) 
    
    languageSimilarityScore = 0
    movie1Langs = [lang.strip() for lang in str(movie1["Language"]).split(",")]
    movie2Langs = [lang.strip() for lang in str(movie2["Language"]).split(",")]
    commonLang = [lang for lang in movie1Langs if lang in movie2Langs]
    if len(commonLang) > 0:
        languageSimilarityScore = 1
    
    ratingSimilarityScore = 1 - np.divide(math.fabs(movie1Rating - movie2Rating), 9)
    plotSimilarityScore = calculatePlotSimilarity(movie1["Plot"], movie2["Plot"])
    
    #print "genresSimilarityScore: " + str(genresSimilarityScore)
    #print "plotSimilarityScore: " + str(plotSimilarityScore)
    #print "ratingSimilarityScore: " + str(ratingSimilarityScore)
    #print "actorsSimilarityScore: " + str(actorsSimilarityScore)
    #print "languageSimilarityScore: " + str(languageSimilarityScore)
    return (genresSimilarityScore + plotSimilarityScore + ratingSimilarityScore + actorsSimilarityScore + languageSimilarityScore) / 5.0

def buildSimilarityMatrix(startIndex, endIndex, moviesData):
    columns = {"movie_id":[]}
    index1 = startIndex
    while index1 < len(moviesData) and index1 < endIndex:
        print index1
        row1 = moviesData.iloc[index1]
        columns["movie_id"].append(row1["movie_id"])
        for index2, row2  in moviesData.iterrows():
            
            key1 = str(row2["movie_id"]) + "<->" + str(row1["movie_id"])
            key2 = str(row1["movie_id"]) + "<->" + str(row2["movie_id"])
            
            if index2 % 100 == 0:
                print key2
            
            if key1 not in cache.keys() and key2 not in cache.keys():
                similarity = computeMovieSimilarity(row1, row2)
                cache[key1] = similarity
                cache[key2] = similarity
            elif key1 in cache.keys():
                similarity = cache[key1]
            else:
                similarity = cache[key2]
                
            if row2["movie_id"] not in columns.keys():
                columns[row2["movie_id"]] = []
            columns[row2["movie_id"]] = similarity
        index1 += 1
    print "**********************"
    return pd.DataFrame.from_dict(columns)

def main():
    moviesDataset = "../resources/movielens-100k-dataset/modified-u.item.csv"
    moviesData = pd.read_csv(moviesDataset, dtype=object)

    pool = ThreadPool(processes=20)
    incrementFactor = 10#len(moviesData)/1000
    startIndex = 0
    endIndex = startIndex + incrementFactor
    
    results = []
    while startIndex < len(moviesData):
        async_result = pool.apply_async(buildSimilarityMatrix, (startIndex, endIndex, moviesData))
        results.append((async_result, startIndex, endIndex))
        startIndex = endIndex
        endIndex = startIndex + incrementFactor
        
    parentDf = None
    flag = True
    for result in results:
        df = result[0].get()
        if parentDf is None:
            parentDf = df
        else:
            parentDf = pd.concat([parentDf, df])
            if flag:
                parentDf.to_csv("../resources/movielens-100k-dataset/similarities1.csv", sep=',', index=False, encoding='utf-8')
                flag = False
    parentDf.to_csv("../resources/movielens-100k-dataset/similarities.csv", sep=',', index=False, encoding='utf-8')

if __name__ == "__main__":
    main()