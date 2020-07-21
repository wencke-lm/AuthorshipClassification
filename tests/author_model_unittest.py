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

from author_model import AuthorModel

class OtherFeatureExtractionFileTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = AuthorModel()
        model.train(os.path.join("data", "frozen", "let_it_go_frozen.txt"))
        cls.features = model.features

    def test_average_sentence_length(self):
        self.assertEqual(self.features["average_sen_length"], 8.222222,
                         msg="\nCalculated average sentence length(left) "
                             "does not align with expectations(right).")

    def test_sentence_length_distribution(self):
        self.assertEqual((self.features["sen_len_0-5"], self.features["sen_len_5-20"]),
                         (0.222222, 0.777778), 
                          msg="\nCalculated sentence distribution over lengths 0-5 "
                             "and 5-20(above) does no align with expectations(below).")

    def test_average_word_length(self):
        self.assertEqual(self.features["average_word_length"], 3.445946,
                         msg="\nCalculated average word length(left) "
                             "does not align with expectations(right).")

    def test_common_word_count(self):
        self.assertEqual((self.features["do"], self.features["see"], self.features["the"]),
                         (5.405405, 1.351351, 4.054054),
                          msg="\nCalculated frequency per 100 words(above) of 'do'(left), "
                              "'see'(middle) or 'the'(right) does not "
                              "align with expectations(below).")

    def test_trigram_count(self):
        self.assertEqual((self.features["['RB', 'DT', 'NN']"], self.features["['VBP', 'RB', 'VB']"],
                          self.features["['RB', 'PRP', 'VBP']"]), (1.388889, 5.555556, 1.388889),
                         msg="\nCalculated frequency per 100 words for the first(left), "
                             "most frequent(middle) or last(right) trigram does not align "
                             "with expectations(below).")

    def test_short_file(self):  # with exactly 3 words
        filename = os.path.join("data", "short_file.txt")
        model = AuthorModel()
        model.train(filename)
        self.assertTrue(True, msg="An exception was raised when parsing a file with two words. "
                                  "Most likely due to trigram generation.")


class OtherFeatureExtractionDirectoryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = AuthorModel()
        model.train(os.path.join("data", "frozen"))
        cls.features = model.features

    def test_average_sentence_length(self):
        self.assertEqual(self.features["average_sen_length"], 8.846154,
                         msg="\nCalculated average sentence length(left) "
                             "does not align with expectations(right).")

    def test_sentence_length_distribution(self):
        self.assertEqual((self.features["sen_len_0-5"], self.features["sen_len_5-20"]),
                         (0.153846, 0.846154),
                          msg="\nCalculated sentence distribution over lengths 0-5 "
                              "and 5-20(above) does no align with expectations(below).")

    def test_average_word_length(self):
        self.assertEqual(self.features["average_word_length"], 3.460870,
                         msg="\nCalculated average word length(left) "
                             "does not align with expectations(right).")

    def test_common_word_count(self):
        self.assertEqual((self.features["do"], self.features["see"], self.features["the"]),
                         (4.347826, 0.869565, 2.608696),
                          msg="\nCalculated frequency per 100 words(above) of 'do'(left), "
                              "'see'(middle) or 'the'(right) does not "
                              "align with expectations(below).")

    def test_trigram_count(self):
        self.assertEqual((self.features["['PRP', 'MD', 'VB']"], self.features["[',', 'VBP', 'RB']"],
                          self.features["['RB', 'PRP', 'VBP']"]), (1.801802, 2.702703, 0.900901),
                         msg="\nCalculated frequency per 100 words for the first(left), "
                             "most frequent(middle) or last(right) trigram does not align "
                             "with expectations(below).")


if __name__ == "__main__":
    unittest.main()