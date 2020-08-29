# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 19/08/2020
# Python 3.7.3
# Windows 8
"""author_model.py testcases."""

import json
import logging
import os
import re
import unittest

from lib.author_model import AuthorModel, LOG
from lib.errors import ScarceDataError


LOG.setLevel(logging.CRITICAL)


class IOInteractionTestCase(unittest.TestCase):
    def test_identity_of_profile_when_originally_trained_as_when_loaded(self):
        trained = AuthorModel.train(os.path.join("tests", "data", "frozen",
                                                 "let_it_go_frozen.txt"))
        loaded = AuthorModel.read_json(os.path.join("tests", "data", "elsa.json"))
        self.assertEqual(trained, loaded)

    def test_preprocessing_a_folder(self):
        result = True
        AuthorModel.preprocess(os.path.join("tests", "data", "raw_data"),
                               os.path.join("tests", "data", "temp"))
        for file in os.listdir(os.path.join("tests", "data", "raw_data")):
            org = os.path.join("tests", "data", "preprocessed_data", file)
            new = os.path.join("tests", "data", "temp", file)
            if os.path.isfile(new):
                with open(org, 'r', encoding='utf-8') as org_file:
                    with open(new, 'r', encoding='utf-8') as new_file:
                        if org_file.read() != new_file.read():
                            result = False
                os.remove(new)
            else:
                result = False
        os.rmdir(os.path.join("tests", "data", "temp"))
        self.assertTrue(result)

    def test_processing_with_json_hook(self):
        data = json.loads('{"1": 27, "5": 42, "2": 80}',
                          object_hook=AuthorModel._objectkeys_to_ints)
        self.assertEqual(data, {1: 27, 5: 42, 2: 80})

    def test_reading_in_corrupted_json_missing_value(self):
        with self.assertRaises(ValueError) as exc:
            AuthorModel.read_json(os.path.join("tests", "data", "corrupted_file2.json"))
        self.assertEqual(str(exc.exception),
                         "Wrong format; make sure to load a JSON-array of length six.")

    def test_reading_in_corrupted_json_wrong_type(self):
        with self.assertRaises(TypeError) as exc:
            AuthorModel.read_json(os.path.join("tests", "data", "corrupted_file1.json"))
        self.assertEqual(str(exc.exception),
                         "The loaded array has to contain a number at last position.")

    def test_training_on_directory_without_files(self):
        with self.assertRaises(FileNotFoundError):
            AuthorModel.train(os.path.join("tests", "data", "dir_without_files"))


class FeatureExtractionFileTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = AuthorModel.train(os.path.join("tests", "data", "frozen",
                                               "let_it_go_frozen.txt"))
        cls.features = model.normalized_feature_vector()

    def test_lemmatization(self):
        lemma = AuthorModel._get_lemma("Did", "VBD")
        self.assertEqual(lemma, "do",
                         msg="Expected the lemma of 'Did' to be 'do'.")

    def test_punctuation_normalized(self):
        self.assertAlmostEqual(self.features[","], 0.56, places=2)

    def test_relative_frequency_of_frequent_word_know(self):
        self.assertAlmostEqual(self.features["know"], 0.0385, places=4)

    def test_training_on_too_short_file(self):
        with self.assertRaises(ScarceDataError):
            AuthorModel.train(os.path.join("tests", "data", "short_file.txt"))

    def test_trigram_frequencies_adding_up_to_one(self):
        trigram_freqs = [self.features[key]
                         for key in self.features
                         if re.match(r"\[.*\]$", key)]
        self.assertAlmostEqual(sum(trigram_freqs), 1, places=4)

    def test_word_length_normalized(self):
        self.assertAlmostEqual(self.features["<w3>"], 0.25, places=2)


class FeatureExtractionDirectoryTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model = AuthorModel.train(os.path.join("tests", "data", "frozen"))
        cls.features = model.normalized_feature_vector()

    def test_relative_frequency_of_frequent_word_do(self):
        self.assertAlmostEqual(self.features["do"], 0.0420, places=4)

    def test_sentence_length_stdev(self):
        self.assertAlmostEqual(self.features["<stdev_sent_len>"], 2.4099, places=4)

    def test_trigram_frequencies_adding_up_to_one(self):
        trigram_freqs = [self.features[key]
                         for key in self.features
                         if re.match(r"\[.*\]$", key)]
        self.assertAlmostEqual(sum(trigram_freqs), 1, places=4)

    def test_word_length_mean(self):
        self.assertAlmostEqual(self.features["<mean_word_len>"], 3.6449, places=4)
