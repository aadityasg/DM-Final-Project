import json
import urllib

OMDB_URL = "http://www.omdbapi.com/?type=movie&r=json&plot=full&i=tt{0}"

def getMovieInformation(imdbId=None):
    url = OMDB_URL.format(imdbId)
    response = urllib.urlopen(url)
    data = json.loads(response.read())
    return data

#imdbInfo = getMovieInformation("0242653")
#print json.dumps(imdbInfo, indent=4, sort_keys=True)