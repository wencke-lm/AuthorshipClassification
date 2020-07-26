# -*- coding: utf-8 -*-

# Wencke Liermann
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 21/07/2020
# Python 3.7.3
# Windows 8
"""author_ident.py testcases."""

from unittest import mock
import os
import unittest
import sys

root_dir = os.path.dirname(os.path.dirname(__file__))
os.chdir(root_dir)
sys.path.append(root_dir)

from scripts.author_model import AuthorModel
from scripts.author_ident import AuthorIdent
from scripts.errors import *


# as far as feasible mocking was used to prevent dependencies between testcases

class AccuracyTestCase(unittest.TestCase):
    @mock.patch("scripts.author_ident.AuthorIdent", autospec=True)
    def test_accuracy_calculation(self, mock_author_ident):
        mock_author_ident.classify.side_effect = ["Jane Austen", "George Eliot", "George Eliot",
                                                  "Charlotte Bronte", "Charlotte Bronte"]
        input_vec = [("", "Jane Austen"), ("", "Jane Austen"),("","George Eliot") ,
                     ("", "Jane Austen"), ("", "Charlotte Bronte")]
        self.assertEqual(AuthorIdent.accuracy(mock_author_ident, input_vec), 0.6,
                         msg="Calculated accuracy(left) does not align with expected(right).")

    @mock.patch("scripts.author_ident.AuthorIdent", autospec=True)
    def test_empty_input_vec(self, mock_author_ident):
        mock_author_ident.classify.side_effect = []
        input_vec = []
        with self.assertRaises(ValueError, msg="Expected ValueError to be raised."):
            AuthorIdent.accuracy(mock_author_ident, input_vec)


class InitTestCase(unittest.TestCase):
    @classmethod
    @mock.patch("scripts.author_ident.AuthorModel", features={}, autospec=True)
    def setUpClass(cls, mock_author_model):
        mock_author_model.read_csv.return_value = mock_author_model
        cls.classifier = AuthorIdent(os.path.join("data", "frozen_catalog.csv"))

    # two asserts but ony functionality: checking a freshly created catalog
    def test_creating_new_classifier(self):
        with AuthorIdent("empty_catalog.csv") as classifier:
            self.assertTrue(os.path.isfile("empty_catalog.csv"),
                            msg="Didn't create new catalog.")
            with open("empty_catalog.csv", 'r', encoding='utf-8') as file_in:
                self.assertEqual(file_in.readlines(), ["author_name\tpretrained_model\n"],
                                 msg="Newly created catalog is malformed.")

    def test_loading_file_that_is_no_catalog(self):
        with self.assertRaises(MalformedCatalogError,
                               msg="Expected MalformedCatalogError to be raised."):
            AuthorIdent(os.path.join("data", "short_file.txt"))

    def test_variable_catalog_content_instantiated(self):
        self.assertEqual(self.classifier.catalog_content,
                         {"elsa":os.path.join("data","elsa.csv")})

    def test_variable_profiles_instantiated(self):
        self.assertIn("elsa", self.classifier.profiles)


class TrainTestCase(unittest.TestCase):
    @classmethod
    @mock.patch("scripts.author_ident.AuthorIdent", catalog=os.path.join("data","elsa.csv"),
                catalog_content={"elsa":os.path.join("data","elsa.csv")}, 
                profiles = {"elsa": None}, autospec=True)
    @mock.patch("scripts.author_ident.AuthorModel", features={}, autospec=True)
    def setUpClass(cls, mock_author_model, mock_author_ident):
        # mock AuthorModel methods
        mock_author_model.train.return_value = mock_author_model
        mock_author_model.write_csv.return_value = None
        # prevent file system from being touched
        with mock.patch('scripts.author_ident.open', mock.mock_open()) as filesys_mock:
            cls.filesys_mock = filesys_mock
            cls.mock_author_model = mock_author_model
            cls.mock_author_ident = mock_author_ident
            AuthorIdent.train(cls.mock_author_ident, "elsa2",
                              os.path.join("data", "frozen", "into_the_unknown_frozen.txt"))

    def test_catalog_file_updated(self):
        self.filesys_mock().write.assert_called_with("elsa2\t" + os.path.join("data", "elsa2.csv\n"))

    def test_profile_file_created(self):
        self.mock_author_model.write_csv.assert_called_with(os.path.join("data", "elsa2.csv"))

    def test_training_for_an_existing_author(self):
        with self.assertRaises(ExistingAuthorError,
                               msg="Expected ExistingAuthorError to be raised."):
            AuthorIdent.train(self.mock_author_ident,"elsa",
                              os.path.join("data", "frozen", "into_the_unknown_frozen.txt"))

    def test_variable_profiles_updated(self):
        self.assertEqual(list(self.mock_author_ident.profiles.keys()), ["elsa", "elsa2"])

    def test_variable_catalog_content_updated(self):
        self.assertEqual(self.mock_author_ident.catalog_content,
                         {"elsa":os.path.join("data","elsa.csv"),
                          "elsa2":os.path.join("data","elsa2.csv")})


class ForgetTestCase(unittest.TestCase):
    @classmethod
    @mock.patch("scripts.author_ident.os")
    @mock.patch("scripts.author_ident.AuthorIdent", catalog=os.path.join("data","elsa.csv"),
                catalog_content={"elsa":os.path.join("data","elsa.csv")}, 
                profiles = {"elsa": None}, autospec=True)
    def setUpClass(cls, mock_author_ident, mock_os):
        # prevent file system from being touched
        with mock.patch('scripts.author_ident.open', mock.mock_open()) as filesys_mock:
            cls.filesys_mock = filesys_mock
            cls.mock_author_ident = mock_author_ident
            cls.mock_os = mock_os
            AuthorIdent.forget(cls.mock_author_ident, "elsa")

    def test_catalog_file_updated(self):
        self.filesys_mock().write.assert_called_with("author_name\tpretrained_model\n")

    def test_deleting_a_not_existing_author(self):
        with self.assertRaises(NotExistingAuthorError,
                               msg="Expected NotExistingAuthorError to be raised."):
            AuthorIdent.forget(self.mock_author_ident, "elsa2")

    def test_profile_file_deleted(self):
        self.mock_os.remove.assert_called_with(os.path.join("data", "elsa.csv"))

    def test_variable_profiles_updated(self):
        self.assertEqual(self.mock_author_ident.profiles, {})

    def test_variable_catalog_content_updated(self):
        self.assertEqual(self.mock_author_ident.catalog_content, {})


class ClassifyTestCase(unittest.TestCase):
    @mock.patch("scripts.author_ident.AuthorModel", autospec=True)
    def test_returned_best_match(self, author_model):
        with mock.patch("scripts.author_ident.AuthorIdent", catalog="",
                        catalog_content={"author1":"author1.csv", "author2":"author2.csv"},
                        profiles = {"author1":author_model, "author2":author_model},
                        autospec=True) as author_ident:
            author_ident.train.return_value = author_model
            author_ident._simil.side_effect = [2.5, 1.6]
            self.assertEqual(AuthorIdent.classify(author_ident, ""), "author2")

    @mock.patch("scripts.author_ident.AuthorIdent", catalog_content={},catalog="", autospec=True)
    def test_too_small_catalog(self, mock_author_ident):
        with self.assertRaises(NotEnoughAuthorsError,
                               msg="Expected NotEnoughAuthorsError to be raised."):
            AuthorIdent.classify(mock_author_ident, "")

    @mock.patch("scripts.author_ident.AuthorModel", features={'i':0.5, 'the':0.5})
    @mock.patch("scripts.author_ident.AuthorModel", features={'do': 0.8, 'the':0.2})
    def test_calculated_similarity_score(self, mock_unk_author, mock_knw_author):
        self.assertEqual(AuthorIdent._simil(mock_knw_author, mock_unk_author), 1.6,
                         msg="\nCalculated similarity score(left) "
                             "does not align with expected(right).")

if __name__ == "__main__":
    unittest.main()