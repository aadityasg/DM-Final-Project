import nltk, string
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import WordNetLemmatizer as wnl

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt') # if necessary...


def stem_tokens(tokens):
    stop = nltk.corpus.stopwords.words('english')
    punctuation = u",.;:'()"
    return [wnl().lemmatize(item.strip(punctuation)) for item in tokens if item not in stop]

def getNormalizeTokens(text):
    remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
    x = stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))
    return x

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

def calculatePlotSimilarity(plot1, plot2):
    plot1 = plot1.decode('unicode-escape')
    plot2 = plot2.decode('unicode-escape')
    
    plot1_tokenList = getNormalizeTokens(plot1)
    plot2_tokenList = getNormalizeTokens(plot2)
    
    tokensCounter =  getTokenOccuringOnce(plot1_tokenList, plot2_tokenList)
    
    plot1_filteredTokens = [item for item in plot1_tokenList if tokensCounter[item] > 1]
    plot2_filteredTokens = [item for item in plot2_tokenList if tokensCounter[item] > 1]

    vect = TfidfVectorizer(min_df=1)
    tfidf = vect.fit_transform([" ".join(plot1_filteredTokens), " ".join(plot2_filteredTokens)])
    sim = (tfidf * tfidf.T).A
    if sim[0][1] > 1:
        return 1
    return sim[0][1]


#print calculatePlotSimilarity(str("the ohio state university is a public university").decode('unicode-escape'), str("University of ohio state is a public university").decode('unicode-escape'))