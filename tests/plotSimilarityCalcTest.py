import unittest
import sys
sys.path.append('../src')
from PlotSimilarityCalcHelper import calculatePlotSimilarity

class plotSimilarityCalTest(unittest.TestCase):
    
    def testSimilarPlots(self):
        plot1 = "The Ohio State University is a public university"
        plot2 = "State university of Ohio is publicly funded university"
        similarity = calculatePlotSimilarity(plot1, plot2)
        expected = 1
        self.assertEqual(similarity, expected, "Expected:{} and Actual:{} similarities are not same!!".format(expected, similarity))

def main():
    unittest.main()

if __name__ == "__main__":
    main()