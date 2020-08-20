# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 19/08/2020
# Python 3.7.3
# Windows 8
"""author_ident.py testcases."""

import logging
import os
import sys
import unittest
from unittest import mock  # to prevent dependencies on the AuthorModel class

root_dir = os.path.dirname(os.path.dirname(__file__))
os.chdir(root_dir)
sys.path.append(root_dir)

from scripts.author_ident import AuthorIdent, LOG
from scripts.errors import CatalogError


LOG.setLevel(logging.CRITICAL)


class AccuracyTestCase(unittest.TestCase):
    @mock.patch("scripts.author_ident.AuthorIdent", autospec=True)
    def test_accuracy_calculation(self, mock_author_ident):
        mock_author_ident.classify.side_effect = ["Jane Austen", "George Eliot", "George Eliot",
                                                  "Charlotte Bronte", "Charlotte Bronte"]
        input_vec = [("", "Jane Austen"), ("", "Jane Austen"), ("", "George Eliot"),
                     ("", "Jane Austen"), ("", "Charlotte Bronte")]
        self.assertEqual(AuthorIdent.accuracy(mock_author_ident, input_vec), 0.6)

    @mock.patch("scripts.author_ident.AuthorIdent", autospec=True)
    def test_empty_input_vec(self, mock_author_ident):
        with self.assertRaises(ValueError, msg="ValueError should have been raised."):
            AuthorIdent.accuracy(mock_author_ident, [])


class ClassifyTestCase(unittest.TestCase):
    def test_calculated_similarity_score(self):
        self.assertEqual(AuthorIdent._simil({'i': 0.5, "<mtld_score>": 50},
                                            {'i': 0.8, "<stdev_word_len>": 1.5}), 0.875)

    @mock.patch("scripts.author_ident.AuthorIdent", catalog="catalog.txt",
                catalog_content={"author1": "author1.csv", "author2": "author2.csv"},
                profiles={"author1": {}, "author2": {}}, autospec=True)
    @mock.patch("scripts.author_ident.AuthorModel", autospec=True)
    def test_returned_best_match(self, mock_author_model, mock_author_ident):
        mock_author_ident._simil.side_effect = [2.5, 1.6]
        self.assertEqual(AuthorIdent.classify(mock_author_ident, ""), "author2")

    @mock.patch("scripts.author_ident.AuthorIdent", catalog="catalog.txt",
                catalog_content={}, autospec=True)
    def test_too_small_catalog(self, mock_author_ident):
        with self.assertRaises(CatalogError):
            AuthorIdent.classify(mock_author_ident, "")


class ForgetTestCase(unittest.TestCase):
    @classmethod
    @mock.patch("scripts.author_ident.os")
    @mock.patch("scripts.author_ident.AuthorIdent", catalog="catalog.txt",
                catalog_content={"author1": "author1.csv"},
                profiles={"author1": {}}, autospec=True)
    def setUpClass(cls, mock_author_ident, mock_os):
        # prevent file system from being touched
        with mock.patch('scripts.author_ident.open', mock.mock_open()) as filesys_mock:
            cls.filesys_mock = filesys_mock
            cls.mock_author_ident = mock_author_ident
            cls.mock_os = mock_os
            AuthorIdent.forget(cls.mock_author_ident, "author1")

    def test_catalog_file_updated(self):
        self.filesys_mock().write.not_called()

    def test_deleting_a_not_existing_author(self):
        with self.assertRaises(CatalogError):
            AuthorIdent.forget(self.mock_author_ident, "author2")

    def test_profile_file_deleted(self):
        self.mock_os.remove.assert_called_with("author1.csv")

    def test_variable_catalog_content_updated(self):
        self.assertEqual(self.mock_author_ident.catalog_content, {})

    def test_variable_profiles_updated(self):
        self.assertEqual(self.mock_author_ident.profiles, {})


class InitTestCase(unittest.TestCase):
    @classmethod
    @mock.patch("scripts.author_ident.AuthorModel", autospec=True)
    def setUpClass(cls, mock_author_model):
        mock_author_model.read_csv.return_value = mock_author_model
        cls.classifier = AuthorIdent(os.path.join("tests", "data", "frozen_catalog.csv"))

    def test_creating_new_classifier(self):
        with mock.patch('scripts.author_ident.open', mock.mock_open()) as filesys_mock:
            AuthorIdent("empty_catalog.csv")
            filesys_mock.assert_called_once()

    def test_loading_corrupted_catalog(self):
        with self.assertLogs(LOG, level='WARNING') as logger:
            AuthorIdent(os.path.join("tests", "data", "corrupted_catalog.txt"))
        msg1 = ("WARNING:scripts.author_ident:Ignored line 3; could not open the file "
                "'tests/data/anna.csv' supposed to contain the pretrained model.")
        msg2 = "WARNING:scripts.author_ident:Ignored line 1; missing column."
        # two asserts but actually only one assertion that logging works as expected
        self.assertIn(msg1, logger.output)
        self.assertIn(msg2, logger.output)

    def test_variable_catalog_content_instantiated(self):
        self.assertEqual(self.classifier.catalog_content,
                         {"elsa": "tests/data/elsa.csv"})

    def test_variable_profiles_instantiated(self):
        self.assertIn("elsa", self.classifier.profiles)


class TrainTestCase(unittest.TestCase):
    @classmethod
    @mock.patch("scripts.author_ident.AuthorIdent", catalog="catalog.txt",
                catalog_content={"author1": "author1.csv"},
                profiles={"author1": {}}, autospec=True)
    @mock.patch("scripts.author_ident.AuthorModel", autospec=True)
    def setUpClass(cls, mock_author_model, mock_author_ident):
        # mock AuthorModel methods
        mock_author_model.train.return_value = mock_author_model
        mock_author_model.write_csv.return_value = None
        # prevent file system from being touched
        with mock.patch('scripts.author_ident.open', mock.mock_open()) as filesys_mock:
            cls.filesys_mock = filesys_mock
            cls.mock_author_model = mock_author_model
            cls.mock_author_ident = mock_author_ident
            AuthorIdent.train(cls.mock_author_ident, "author2", "author2.txt")

    def test_catalog_file_updated(self):
        self.filesys_mock().write.assert_called_with("author2\tauthor2.csv\n")

    def test_profile_file_created(self):
        self.mock_author_model.write_csv.assert_called_with("author2.csv")

    def test_training_for_an_existing_author(self):
        with self.assertRaises(CatalogError):
            AuthorIdent.train(self.mock_author_ident, "author1", "author1.txt")

    def test_variable_catalog_content_updated(self):
        self.assertEqual(self.mock_author_ident.catalog_content,
                         {"author1": "author1.csv", "author2": "author2.csv"})

    def test_variable_profiles_updated(self):
        self.assertEqual(list(self.mock_author_ident.profiles.keys()), ["author1", "author2"])


if __name__ == "__main__":
    unittest.main()
