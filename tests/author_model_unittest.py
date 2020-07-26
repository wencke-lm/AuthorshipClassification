# -*- coding: utf-8 -*-

# Wencke Liermann
# Universität Potsdam
# Bachelor Computerlinguistik

# 21/07/2020
# Python 3.7.3
# Windows 8
"""otherfeatures.py testcase"""

import os
import unittest
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
os.chdir(root_dir)
sys.path.append(root_dir)

from scripts.author_model import AuthorModel

class FeatureExtractionFileTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = AuthorModel.train(os.path.join("data", "frozen", "let_it_go_frozen.txt"))
        cls.features = model.features

    def test_average_sentence_length(self):
        self.assertEqual(self.features["average_sen_length"], 0.0822222222,
                         msg="\nCalculated average sentence length(left) "
                             "does not align with expectations(right).")

    def test_sentence_length_distribution(self):
        self.assertEqual((self.features["<sen_len_0-5>"], self.features["<sen_len_5-20>"]),
                         (0.2222222222, 0.7777777778), 
                          msg="\nCalculated sentence distribution over lengths 0-5 "
                             "and 5-20(above) does no align with expectations(below).")

    def test_average_word_length(self):
        self.assertEqual(self.features["average_word_length"], 0.3445945946,
                         msg="\nCalculated average word length(left) "
                             "does not align with expectations(right).")

    def test_common_word_count(self):
        self.assertEqual((self.features["do"], self.features["see"], self.features["the"]),
                         (0.0540540541, 0.027027027, 0.0405405405),
                          msg="\nCalculated frequency per 100 words(above) of 'do'(left), "
                              "'see'(middle) or 'the'(right) does not "
                              "align with expectations(below).")

    def test_trigram_count(self):
        self.assertEqual((self.features["['RB', 'DT', 'NN']"], self.features["['VBP', 'RB', 'VB']"],
                          self.features["['RB', 'PRP', 'VBP']"]), (0.0138888889, 0.0555555556,
                          0.0138888889), msg="\nCalculated frequency per 100 words "
                             "for the first(left), most frequent(middle) or last(right) "
                             "trigram does not align with expectations(below).")

    @unittest.expectedFailure
    def test_short_file(self):  # with exactly 2 words
        filename = os.path.join("data", "short_file.txt")
        with self.assertRaises(ZeroDivisionError,
                               msg="An exception was raised when parsing a file with "
                               "two words. Most likely due to trigram generation."):
            model = AuthorModel.train(filename)



class FeatureExtractionDirectoryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = AuthorModel.train(os.path.join("data", "frozen"))
        cls.features = model.features

    def test_average_sentence_length(self):
        self.assertEqual(self.features["average_sen_length"], 0.0884615385,
                         msg="\nCalculated average sentence length(left) "
                             "does not align with expectations(right).")

    def test_sentence_length_distribution(self):
        self.assertEqual((self.features["<sen_len_0-5>"], self.features["<sen_len_5-20>"]),
                         (0.1538461538, 0.8461538462),
                          msg="\nCalculated sentence distribution over lengths 0-5 "
                              "and 5-20(above) does no align with expectations(below).")

    def test_average_word_length(self):
        self.assertEqual(self.features["average_word_length"], 0.3460869565,
                         msg="\nCalculated average word length(left) "
                             "does not align with expectations(right).")

    def test_common_word_count(self):
        self.assertEqual((self.features["do"], self.features["see"], self.features["the"]),
                         (0.0434782609, 0.0173913043, 0.0260869565),
                          msg="\nCalculated frequency per 100 words(above) of 'do'(left), "
                              "'see'(middle) or 'the'(right) does not "
                              "align with expectations(below).")

    def test_trigram_count(self):
        self.assertEqual((self.features["['PRP', 'MD', 'VB']"], self.features["[',', 'VBP', 'RB']"],
                          self.features["['RB', 'PRP', 'VBP']"]), (0.018018018, 0.027027027,
                          0.009009009), msg="\nCalculated frequency per 100 words for the "
                             "first(left), most frequent(middle) or last(right) trigram does "
                             "not align with expectations(below).")


if __name__ == "__main__":
    unittest.main()