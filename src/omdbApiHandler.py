import json
import urllib

dummyData = {'Plot': "?", 
            'Rated': '?', 
            'Title': '?', 
            'Ratings': [{'Source': 'Internet Movie Database', 'Value': '?'}, 
                                                    {'Source': 'Rotten Tomatoes', 'Value': '?'}, 
                                                    {'Source': 'Metacritic', 'Value': '?'}], 
            'DVD': '?', 
            'Writer': '?', 
            'Production': '?', 
            'Actors': '?', 
            'Type': 'movie', 
            'imdbVotes': '?', 
            'Director': '?', 
            'Released': '?', 
            'Awards': '?.', 
            'Genre': '?', 
            'imdbRating': '?', 
            'Language': '?', 
            'Country': '?', 
            'imdbID': '?', 
            'Metascore': '?',
            'Year': '?'}


OMDB_URL = "http://www.omdbapi.com/?type=movie&r=json&plot=full&i=tt{0}"

def getMovieInformation(imdbId=None):
    url = OMDB_URL.format(imdbId)
    response = urllib.urlopen(url)
    try:
        data = json.loads(response.read())
    except:
        data = deepcopy(dummyData)
    return data

#imdbInfo = getMovieInformation("0242653")
#print json.dumps(imdbInfo, indent=4, sort_keys=True)