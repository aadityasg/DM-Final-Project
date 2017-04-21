import numpy as np
import math
import pandas as pd
from PlotSimilarityCalcHelper import calculatePlotSimilarity

def computeMovieSimilarity(movie1, movie2):
    print "Comparing:"
    print movie1["title"]
    print movie2["title"]
    
    movie1Rating = float(movie1["imdbRating"])
    movie2Rating = float(movie2["imdbRating"])
    
    actors1 = [actor.strip() for actor in movie1["Actors"].split(",")]
    actors2 = [actor.strip() for actor in movie2["Actors"].split(",")]
    actors = set(actors1)
    actors |= set(actors2)
    commonActors = [actor for actor in actors1 if actor in actors2]
    actorsSimilarityScore = len(commonActors) / float(len(actors)) 
    
    
    genres1 = [genre.strip() for genre in movie1["genres"].split("|")]
    genres2 = [genre.strip() for genre in movie2["genres"].split("|")]
    genres = set(genres1)
    genres |= set(genres2)
    commonGenres = [genre for genre in genres1 if genre in genres2]
    genresSimilarityScore = len(commonGenres) / float(len(genres)) 
    
    languageSimilarityScore = 0
    movie1Langs = [lang.strip() for lang in movie1["Language"].split(",")]
    movie2Langs = [lang.strip() for lang in movie2["Language"].split(",")]
    commonLang = [lang for lang in movie1Langs if lang in movie2Langs]
    if len(commonLang) > 0:
        languageSimilarityScore = 1
    
    ratingSimilarityScore = 1 - np.divide(math.fabs(movie1Rating - movie2Rating), 9)
    plotSimilarityScore = calculatePlotSimilarity(movie1["Plot"], movie2["Plot"])
    
    print "genresSimilarityScore: " + str(genresSimilarityScore)
    print "plotSimilarityScore: " + str(plotSimilarityScore)
    print "ratingSimilarityScore: " + str(ratingSimilarityScore)
    print "actorsSimilarityScore: " + str(actorsSimilarityScore)
    print "languageSimilarityScore: " + str(languageSimilarityScore)

    return (genresSimilarityScore + plotSimilarityScore + ratingSimilarityScore + actorsSimilarityScore + languageSimilarityScore) / 5.0

moviesDataset = "../resources/modified_movies_dataset.csv"
moviesData = pd.read_csv(moviesDataset, dtype=object)

print computeMovieSimilarity(moviesData.iloc[7670], moviesData.iloc[7842])

#print "\n\n\n"
#for index, row  in moviesData.iterrows():
#    print str(index) + ", " + row["title"]