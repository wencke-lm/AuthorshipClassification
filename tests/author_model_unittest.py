# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 19/08/2020
# Python 3.7.3
# Windows 8
"""author_model.py testcases."""

import filecmp
import logging
import os
import re
import sys
import unittest

root_dir = os.path.dirname(os.path.dirname(__file__))
os.chdir(root_dir)
sys.path.append(root_dir)

from scripts.author_model import AuthorModel, LOG
from scripts.errors import ScarceDataError

LOG.setLevel(logging.CRITICAL)

class IOInteractionTestCase(unittest.TestCase):
    def test_training_on_directory_without_files(self):
        with self.assertRaises(FileNotFoundError):
            AuthorModel.train(os.path.join("tests", "data", "dir_without_files"))

    def test_logs_for_reading_in_corrupted_csv_file(self):
        with self.assertLogs(LOG, level='WARNING') as logger:
            AuthorModel.read_csv(os.path.join("tests", "data", "corrupted_file.csv"))
        msg1 = "WARNING:scripts.author_model:ignored line 3; missing column"
        msg2 = "WARNING:scripts.author_model:ignored line 5; not-float value in second column"
        # two asserts but actually only one assertion that logging works as expected
        self.assertIn(msg1, logger.output)
        self.assertIn(msg2, logger.output)

    def test_identity_of_profile_when_originally_trained_as_when_loaded(self):
        trained = AuthorModel.train(os.path.join("tests", "data", "frozen",
                                                 "let_it_go_frozen.txt"))
        loaded = AuthorModel.read_csv(os.path.join("tests", "data", "elsa.csv"))
        self.assertEqual(trained, loaded)

    def test_preprocessing_a_folder(self):
        result = True
        AuthorModel.preprocess(os.path.join("tests", "data", "raw_data"),
                               os.path.join("tests", "data", "temp"))
        for fl in os.listdir(os.path.join("tests", "data", "raw_data")):
            org = os.path.join("tests", "data", "preprocessed_data", fl)
            new = os.path.join("tests", "data", "temp", fl)
            if os.path.isfile(new):
                if not filecmp.cmp(org, new, shallow=False):
                    result = False
                os.remove(new)
            else:
                result = False
        os.rmdir(os.path.join("tests", "data", "temp"))
        self.assertTrue(result)


class FeatureExtractionFileTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AuthorModel.train(os.path.join("tests", "data", "frozen",
                                                   "let_it_go_frozen.txt"))

    def test_training_on_too_short_file(self):
        with self.assertRaises(ScarceDataError):
            AuthorModel.train(os.path.join("tests", "data", "short_file.txt"))

    def test_sentence_length_stdev(self):
        self.assertAlmostEqual(self.model["<stdev_sent_len>"], 2.6822, places=4)

    def test_word_length_mean(self):
        self.assertAlmostEqual(self.model["<mean_word_len>"], 3.6232, places=4)

    def test_relative_frequency_of_frequent_word_know(self):
        self.assertAlmostEqual(self.model["know"], 0.0405, places=4)

    def test_trigram_frequencies_adding_up_to_one(self):
        trigram_freqs = [self.model[key]
                         for key in self.model
                         if re.match(r"\[.*\]$", key)]
        self.assertAlmostEqual(sum(trigram_freqs), 1, places=4)

    def test_lemmatization(self):
        lemma = AuthorModel._get_lemma("Did", "VBD")
        self.assertEqual(lemma, "do",
                         msg="Expected the lemma of 'Did' to be 'do'.")


class FeatureExtractionDirectoryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = AuthorModel.train(os.path.join("tests", "data", "frozen"))

    def test_relative_frequency_of_frequent_word_do(self):
        self.assertAlmostEqual(self.model["do"], 0.0435, places=4)

    def test_sentence_length_stdev(self):
        self.assertAlmostEqual(self.model["<stdev_sent_len>"], 2.6409, places=4)

    def test_word_length_mean(self):
        self.assertAlmostEqual(self.model["<mean_word_len>"], 3.6449, places=4)

    def test_trigram_frequencies_adding_up_to_one(self):
        trigram_freqs = [self.model[key]
                         for key in self.model
                         if re.match(r"\[.*\]$", key)]
        self.assertAlmostEqual(sum(trigram_freqs), 1, places=4)


if __name__ == "__main__":
    unittest.main()
