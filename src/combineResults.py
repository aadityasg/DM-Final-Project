import pandas as pd

def combine():
	similarityData = "../resources/movielens-100k-dataset/similarities-inc-{0}.csv"
	parentDf = pd.read_csv("../resources/movielens-100k-dataset/similarities-inc-parent.csv")
	start = 191
	while start<= 1681:
		print start
		df = pd.read_csv(similarityData.format(start), dtype=object)
		parentDf = pd.concat([parentDf, df])
		start += 10
	parentDf.to_csv("../resources/movielens-100k-dataset/similarities-final.csv")
	return None

print "Here"
combine()
