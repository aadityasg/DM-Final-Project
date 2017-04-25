from nltk import ne_chunk, pos_tag
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import WordNetLemmatizer as wnl
from nltk.tree import Tree
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
import pandas as pd
import hashlib

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt') # if necessary
nltk.download('averaged_perceptron_tagger')
nltk.download("maxent_ne_chunker")
nltk.download("words")
wn.ensure_loaded() 
    
def stem_tokens(tokens):
    stop = nltk.corpus.stopwords.words('english')
    punctuation = u",.;:'()"
    return [wnl().lemmatize(item.strip(punctuation)) for item in tokens if item not in stop]

def getNormalizeTokens(text):
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    x = stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
    return x

def getNameList(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in chunked:
            if type(i) == Tree:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            elif current_chunk:
                named_entity = " ".join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
    return continuous_chunk

def getTokenOccuringOnce(list1, list2):
    wordCounter = {}
    for word in list1:
        if word not in wordCounter:
            wordCounter[word] = 0
        wordCounter[word] += 1
        
    for word in list2:
        if word not in wordCounter:
            wordCounter[word] = 0
        wordCounter[word] += 1
    return wordCounter

cache = {}

moviesDataset = "../resources/movielens-100k-dataset/modified-u.item.csv"
moviesData = pd.read_csv(moviesDataset, dtype=object)

for index2, movie  in moviesData.iterrows():
    plot = movie["Plot"]
    plotDigest = hashlib.md5(str(plot)).hexdigest()
    plot = unicode(str(plot), 'ascii', 'ignore') #plot1.encode('utf-8')
    
    plot_tokenList = getNormalizeTokens(plot)
    
    names = []
    names.extend(getNameList(plot))
    names = [name.lower() for name in names]
    
    tokensCounter =  getTokenOccuringOnce(plot_tokenList, [])
    
    plot_filteredTokens = [item for item in plot_tokenList if tokensCounter[item] > 1 or item in names]
    
    plt = " ".join(plot_filteredTokens)
    
    cache[plotDigest] = plt
    print plotDigest + "  ->  " + plt

def calculatePlotSimilarity(plot1, plot2):
    # More info at:
    # http://nbviewer.jupyter.org/gist/francoiseprovencher/83c595531177ac88e3c0
    
    plot1Digest = hashlib.md5(str(plot1)).hexdigest()
    plot2Digest = hashlib.md5(str(plot2)).hexdigest()
    
    plt1 = cache[plot1Digest]
    plt2 = cache[plot2Digest]
    
    if len(plt1) == 0 or len(plt2) == 0:
        return 0
    
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform([plt1, plt2])
    sim = (tfidf * tfidf.T).A
    if sim[0][1] > 1:
        return 1
    return sim[0][1]

#print calculatePlotSimilarity(str("the ohio state university is a public university where John goes to").decode('unicode-escape'), str("University of ohio state is a public university").decode('unicode-escape'))
