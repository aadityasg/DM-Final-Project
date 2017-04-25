import json
import urllib
from copy import deepcopy
import re
import string
printable = set(string.printable)

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


OMDB_URL = "http://www.omdbapi.com/?type=movie&r=json&plot=full&t={0}&y={1}"

def getMovieInformation(imdbTitle=None):
    found = re.search("(.*) \\((\\d*)\\)", imdbTitle)
    if found is None:
        return dummyData

    title = found.group(1)
    year = found.group(2)
    url = OMDB_URL.format(title, year)
    response = urllib.urlopen(url)
    data = response.read().decode('utf-8', 'ignore')
    data = filter(lambda x: x in printable, data)
    try:
        data = json.loads(data)
    except:
        data = deepcopy(dummyData)
    return data

imdbInfo = getMovieInformation("Toy Story (1995)")
#print imdbInfo
#print json.dumps(imdbInfo, indent=4, sort_keys=True)