import pandas as pd
import omdbApiHandler as omdb
import numpy
from copy import deepcopy
from multiprocessing.pool import ThreadPool

def getMoviesDataWithImdbLink(moviesDataset = "../resources/movies.csv", linksDataset = "../resources/links.csv"):
    moviesData = pd.read_csv(moviesDataset, dtype=object)
    linksData = pd.read_csv(linksDataset, dtype=object)

    moviesData["imdbId"] = pd.Series(linksData["imdbId"], index=linksData.index)
    return moviesData


def getGenres(row, imdbInfo):
    genres = set([x.lower() for x in row["genres"].split("|")])
    if imdbInfo["Genre"] != "?":
        genres.union(set([x.lower() for x in imdbInfo["Genre"].split(", ")]))
    return ", ".join(genres)


def buildInformationArray(moviesData, startIndex, endIndex):
    emptyData = []
    Actors, Language, Plot, Rated, Director = deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData)
    Released, Year, Writer, imdbRating, imdbVotes = deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData) 
    RottenTomatoesRatings, MetacriticRatings = deepcopy(emptyData), deepcopy(emptyData)
    index = startIndex
    counter = 0
    while index < endIndex and index < len(moviesData):
        row = moviesData.iloc[index]
        index += 1
        imdbInfo = omdb.getMovieInformation(row["imdbId"])
        #print json.dumps(imdbInfo, indent=4, sort_keys=True)
        
        counter += 1
        if counter % 50 == 0:
            print counter
        
        
        if "Genre" in imdbInfo.keys():
            row["genres"] = getGenres(row, imdbInfo)
            
        
        if "Actors" in imdbInfo.keys():
            Actors.append(imdbInfo["Actors"])
        else:
            Actors.append("?")
        
        if "Language" in imdbInfo.keys():
            Language.append(imdbInfo["Language"])
        else:
            Language.append("?")
        
        if "Director" in imdbInfo.keys():
            Director.append(imdbInfo["Director"])
        else:
            Director.append("?")
        
        if "Rated" in imdbInfo.keys():
            Rated.append(imdbInfo["Rated"])
        else:
            Rated.append("?")
        
        if "Plot" in imdbInfo.keys():
            Plot.append(imdbInfo["Plot"])
        else:
            Plot.append("?")
        
        if "Released" in imdbInfo.keys():
            Released.append(imdbInfo["Released"])
        else:
            Released.append("?")
        
        if "Year" in imdbInfo.keys():
            Year.append(imdbInfo["Year"])
        else:
            Year.append("?")
        
        if "Writer" in imdbInfo.keys():
            Writer.append(imdbInfo["Writer"])
        else:
            Writer.append("?")
        
        if "imdbRating" in imdbInfo.keys():
            imdbRating.append(imdbInfo["imdbRating"])
        else:
            imdbRating.append("?")
        
        if "imdbVotes" in imdbInfo.keys():
            imdbVotes.append(imdbInfo["imdbVotes"])
        else:
            imdbVotes.append("?")
            
        if "Ratings" in imdbInfo.keys():
            flagRotten = False
            flagMetacritic = False
            for rating in imdbInfo["Ratings"]:
                if rating["Source"] == "Rotten Tomatoes":
                    RottenTomatoesRatings.append(rating["Value"])
                    flagRotten = True
                if rating["Source"] == "Metacritic":
                    MetacriticRatings.append(rating["Value"])
                    flagMetacritic = True
            if not flagRotten:
                RottenTomatoesRatings.append("?")
            if not flagMetacritic:
                MetacriticRatings.append("?")
                
        else:
            RottenTomatoesRatings.append("?")
            MetacriticRatings.append("?")
        
    return (Actors, Language, Plot, Rated, Director, Released, Year, Writer, imdbRating, imdbVotes, RottenTomatoesRatings, MetacriticRatings, index)


def getCompleteMoviesInformation(moviesDataset = "../resources/movies.csv", linksDataset = "../resources/links.csv"):
    moviesData = getMoviesDataWithImdbLink(moviesDataset, linksDataset)
    
    emptyData = numpy.empty_like(moviesData["movieId"])
    emptyData[:] = "?"
    
    ActorsTotal, LanguageTotal, PlotTotal, RatedTotal, DirectorTotal = deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData)
    ReleasedTotal, YearTotal, WriterTotal, imdbRatingTotal, imdbVotesTotal = deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData), deepcopy(emptyData) 
    RottenTomatoesRatingsTotal, MetacriticRatingsTotal = deepcopy(emptyData), deepcopy(emptyData)
    
    pool = ThreadPool(processes=100)
    incrementFactor = len(moviesData)/100
    startIndex = 0
    endIndex = startIndex + incrementFactor
    
    results = []
    while startIndex < len(moviesData):
        
        async_result = pool.apply_async(buildInformationArray, (moviesData, startIndex, endIndex))
        results.append((async_result, startIndex, endIndex))
        startIndex = endIndex
        endIndex = startIndex + incrementFactor
        
        # do some other stuff in the main process
    
    for result in results:
        Actors, Language, Plot, Rated, Director, Released, \
        Year, Writer, imdbRating, imdbVotes, RottenTomatoesRatings, \
        MetacriticRatings, end = result[0].get()
        
        startIndex = result[1]
        
        ActorsTotal[startIndex:end] = Actors
        YearTotal[startIndex:end] = Year
        LanguageTotal[startIndex:end] = Language
        PlotTotal[startIndex:end] = Plot
        RatedTotal[startIndex:end] = Rated
        DirectorTotal[startIndex:end] = Director
        ReleasedTotal[startIndex:end] = Released
        YearTotal[startIndex:end] = Year
        WriterTotal[startIndex:end] = Writer
        imdbRatingTotal[startIndex:end] = imdbRating
        imdbVotesTotal[startIndex:end] = imdbVotes
        RottenTomatoesRatingsTotal[startIndex:end] = RottenTomatoesRatings
        MetacriticRatingsTotal[startIndex:end] = MetacriticRatings
    
    moviesData["serialNo"] = pd.Series([i+1 for i in range(len(ActorsTotal))], index=moviesData.index)
    moviesData["Actors"] = pd.Series(ActorsTotal, index=moviesData.index)
    moviesData["Language"] = pd.Series(LanguageTotal, index=moviesData.index)
    moviesData["Director"] = pd.Series(DirectorTotal, index=moviesData.index)
    moviesData["Rated"] = pd.Series(RatedTotal, index=moviesData.index)
    moviesData["Plot"] = pd.Series(PlotTotal, index=moviesData.index)
    moviesData["Released"] = pd.Series(ReleasedTotal, index=moviesData.index)
    moviesData["Year"] = pd.Series(YearTotal, index=moviesData.index)
    moviesData["Writer"] = pd.Series(WriterTotal, index=moviesData.index)
    moviesData["RottenTomatoesRating"] = pd.Series(RottenTomatoesRatingsTotal, index=moviesData.index)
    moviesData["Metacritic"] = pd.Series(MetacriticRatingsTotal, index=moviesData.index)
    moviesData["imdbRating"] = pd.Series(imdbRatingTotal, index=moviesData.index)
    moviesData["imdbVotes"] = pd.Series(imdbVotesTotal, index=moviesData.index)
    
    moviesData.to_csv("../resources/modified_movies_dataset.csv", sep=',', index=False, encoding='utf-8')
    return moviesData

print getCompleteMoviesInformation()