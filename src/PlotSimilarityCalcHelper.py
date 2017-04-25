from nltk import ne_chunk, pos_tag
import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import WordNetLemmatizer as wnl
from nltk.tree import Tree
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn

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

def calculatePlotSimilarity(plot1, plot2):
    # More info at:
    # http://nbviewer.jupyter.org/gist/francoiseprovencher/83c595531177ac88e3c0
    
    plot1Digest = hashlib.md5(str(plot1)).hexdigest()
    plot2Digest = hashlib.md5(str(plot2)).hexdigest()
    
    if plot1Digest in cache.keys() and plot2Digest in cache.keys():
        plt1 = cache[plot1Digest]
        plt2 = cache[plot2Digest]
    else:
        plot1 = unicode(str(plot1), 'ascii', 'ignore') #plot1.encode('utf-8')
        plot2 = unicode(str(plot2), 'ascii', 'ignore') #plot2.encode('utf-8')
        
        plot1_tokenList = getNormalizeTokens(plot1)
        plot2_tokenList = getNormalizeTokens(plot2)
        
        names = []
        names.extend(getNameList(plot1))
        names.extend(getNameList(plot2))
        
        names = [name.lower() for name in names]
        
        tokensCounter =  getTokenOccuringOnce(plot1_tokenList, plot2_tokenList)
        
        
        
        plot1_filteredTokens = [item for item in plot1_tokenList if tokensCounter[item] > 1 or item in names]
        plot2_filteredTokens = [item for item in plot2_tokenList if tokensCounter[item] > 1 or item in names]
        
        plt1 = " ".join(plot1_filteredTokens)
        plt2 = " ".join(plot2_filteredTokens)
        
        cache[plot1Digest] = plt1
        cache[plot2Digest] = plt2
        
    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform([plt1, plt2])
    sim = (tfidf * tfidf.T).A
    if sim[0][1] > 1:
        return 1
    return sim[0][1]

#print calculatePlotSimilarity(str("the ohio state university is a public university where John goes to").decode('unicode-escape'), str("University of ohio state is a public university").decode('unicode-escape'))
