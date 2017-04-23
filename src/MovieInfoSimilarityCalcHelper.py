import numpy as np
import math
import pandas as pd
from PlotSimilarityCalcHelper import calculatePlotSimilarity
from multiprocessing.pool import ThreadPool

cache = {}


def computeMovieSimilarity(movie1, movie2):
    #print movie1["title"] + " vs " + movie2["title"]
    
    movie1Rating = float(movie1["imdbRating"])
    movie2Rating = float(movie2["imdbRating"])
    
    actors1 = [actor.strip() for actor in str(movie1["Actors"]).split(",")]
    actors2 = [actor.strip() for actor in str(movie2["Actors"]).split(",")]
    actors = set(actors1)
    actors |= set(actors2)
    commonActors = [actor for actor in actors1 if actor in actors2]
    actorsSimilarityScore = len(commonActors) / float(len(actors)) 
    
    
    genres1 = [genre.strip() for genre in str(movie1["genres"]).split("|")]
    genres2 = [genre.strip() for genre in str(movie2["genres"]).split("|")]
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
    columns = {"movieId":[]}
    index1 = startIndex
    while index1 < len(moviesData) and index1 < endIndex:
        print index1
        row1 = moviesData.iloc[index1]
        columns["movieId"].append(row1["movieId"])
        for index2, row2  in moviesData.iterrows():
            key1 = str(row2["movieId"]) + "<->" + str(row1["movieId"])
            key2 = str(row1["movieId"]) + "<->" + str(row2["movieId"])
            print key2
            if key1 not in cache.keys() and key2 not in cache.keys():
                similarity = computeMovieSimilarity(row1, row2)
                cache[key1] = similarity
                cache[key2] = similarity
            elif key1 in cache.keys():
                similarity = cache[key1]
            else:
                similarity = cache[key2]
                
            if row2["movieId"] not in columns.keys():
                columns[row2["movieId"]] = []
            columns[row2["movieId"]] = similarity
        index1 += 1
    print "**********************"
    return pd.DataFrame.from_dict(columns, index=columns["movieId"])

def main():
    moviesDataset = "../resources/modified_movies_dataset.csv"
    moviesData = pd.read_csv(moviesDataset, dtype=object)

    pool = ThreadPool(processes=20)
    incrementFactor = 5#len(moviesData)/1000
    startIndex = 0
    endIndex = startIndex + incrementFactor
    
    results = []
    while startIndex < len(moviesData):
        async_result = pool.apply_async(buildSimilarityMatrix, (startIndex, endIndex, moviesData))
        results.append((async_result, startIndex, endIndex))
        startIndex = endIndex
        endIndex = startIndex + incrementFactor
        
    parentDf = None
    
    for result in results:
        df = result[0].get()
        if parentDf is None:
            parentDf = df
        else:
            parentDf = pd.concat(parentDf, df)
    
    parentDf.to_csv("../resources/sample.csv", sep=',', index=False, encoding='utf-8')

if __name__ == "__main__":
    main()