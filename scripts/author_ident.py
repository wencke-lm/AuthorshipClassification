# -*- coding: utf-8 -*-

# Wencke Liermann
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 21/07/2020
# Python 3.7.3
# Windows 8
"""Classifier for feature-based authorship attribution."""

import csv
import logging
import os

from scripts.author_model import AuthorModel
from scripts.errors import *

LOG = logging.getLogger(__name__)

class AuthorIdent:
    """Collects author profiles in a catalog.

    As part of the project a catalog called 'gutenbergident.txt'
    containing pretrained models for the following _ authors
    can be accessed:


    The trainings data for this catalog has been taken from:
    Lahiri, S. (2014). Complexity of Word Collocation Networks:
    A Preliminary Structural Analysis. In Proceedings of the Student
    Research Workshop at the 14th Conference of the European Chapter
    of the Association for Computational Linguistics (pp. 96–105).
    Association for Computational Linguistics.

    Args:
        catalog(str): Path to a file containing lines of the form
            <author>\t<pretrained model .csv-filename> .

    Attributes:
        catalog(str)
        catalog_content(dict): Maps author names to the filenames
            their pretrained models are saved under.
        profiles(dict): Maps author names to
            their loaded author profiles.
    """
    def __init__(self, catalog):
        self.catalog = catalog
        self.catalog_content = dict()
        self.profiles = dict()
        
        self._read_catalog(catalog)

    def train(self, author_name, source):
        """Add new author profile to classifier.

        Args:
            author_name(str): Name of the newly created class.
            source(str): Path to an utf8-encoded .txt-file.
                Alternatively one can also pass a directory containing
                such files.
                All the files must have already been preprocessed,
                separating tokens with whitespaces and giving each
                sentence a single line. One can use the function
                <AuthorModel.preprocess> to perform this preprocessing.

        Returns:
            None
        """
        if author_name in self.profiles:
            raise ExistingAuthorError(author_name)
        author_pr = AuthorModel.train(source)
        self.profiles[author_name] = author_pr
        filepath = os.path.join(os.path.dirname(self.catalog), author_name)
        if os.path.isfile(filepath + ".csv"):
            i = 2
            while os.path.isfile(filepath + f"({i})"+ ".csv"):
                i += 1
            filepath = filepath + f"({i})"
        author_pr.write_csv(filepath +".csv")
        with open(self.catalog, 'a', encoding='utf-8') as file_out:
            file_out.write(f"{author_name}\t{filepath +'.csv'}\n")
        self.catalog_content[author_name] = filepath + ".csv"

    def forget(self, author_name):
        """Remove author profile from classifier.

        The saved pretrained model of this author will be deleted
        in the process as well.

        Args:
            author_name(str): Name of the class to be deleted.

        Returns:
            None
        """
        if author_name not in self.catalog_content:
            raise NotExistingAuthorError(author_name)
        self.profiles.pop(author_name)
        saved_model = self.catalog_content.pop(author_name)
        if os.path.isfile(saved_model):
            os.remove(saved_model)
        else:
            LOG.warning(f"The pretrained model '{saved_model}' "
                        "of the deleted author couldn't be removed.")
        with open(self.catalog, 'w', encoding='utf-8') as file_out:
            file_out.write("author_name\tpretrained_model\n")
            for author, profile in self.catalog_content.items():
                file_out.write(f"{author}\t{profile}\n")

    def classify(self, source):
        """Perform authorship attribution for given text file.

        Args:
            source(str): Path to an utf8-encoded .txt-file.
                The file must have already been preprocessed,
                separating tokens with whitespaces and giving each
                sentence a single line. One can use the function
                <AuthorModel.preprocess> to perform this preprocessing.

        Returns:
            str: Most likely author for the text.
        """
        if len(self.catalog_content) < 2:
            raise NotEnoughAuthorsError(self.catalog)
        unknown_author_pr = AuthorModel.train(source)
        best_match = (float('inf'), None)
        for known_author, known_author_pr in self.profiles.items():
            diff = self._simil(known_author_pr, unknown_author_pr)
            if diff < best_match[0]:
                best_match = (diff, known_author)
        return best_match[1]

    def accuracy(self, input_vec):
        """Calculate the accuracy on an annotated test set.

        Args:
            input_vec(list<tuple>): Pairs containing the data to
                be classified and its correct class.

        Returns:
            float: Number of correctly annotated texts divided by the
                number of texts inputed.
        """
        if input_vec == []:
            raise ValueError("Accuracy of an empty test set can't be calculated.")
        correct = 0
        for pair in input_vec:
            result = self.classify(pair[0])
            if result == pair[1]:
                correct += 1
        return correct/len(input_vec)

    def destroy(self):
        """Deletes the catalog and all pretrained models linked to it."""
        for author in list(self.catalog_content.keys()):
            self.forget(author)
        if os.path.isfile(self.catalog):
            os.remove(self.catalog)
        else:
            LOG.warning(f"The catalog '{self.catalog}' could not be deleted.")

######################################## private methods ########################################

    def _read_catalog(self, catalog):
        """Load saved classifier."""
        if os.path.isfile(catalog):
            LOG.info(f"Load existing classifier with the catalog '{catalog}' ...")
            with open(catalog, 'r', encoding='utf-8') as file_in:
                csv_reader = csv.DictReader(file_in, delimiter='\t')
                if csv_reader.fieldnames != ["author_name", "pretrained_model"]:
                    raise MalformedCatalogError(catalog)
                for ln, line in enumerate(csv_reader, 1):
                    author, model = line["author_name"], line["pretrained_model"]
                    try:
                        author_pr = AuthorModel.read_csv(model)
                    except FileNotFoundError:
                        LOG.warning(f"Ignored line {ln}; could not open the file "
                                    f"'{model}' supposed to contain the pretrained model.")
                    else:
                        self.profiles[author] = author_pr
                        self.catalog_content[author] = model
                        LOG.info(f"{author}\t{model}")
        else:
            LOG.info(f"Create new classifier with the catalog '{catalog}' ...")
            with open(catalog, 'w', encoding='utf-8') as file_out:
                file_out.write(f"author_name\tpretrained_model\n")

    @staticmethod
    def _simil(known_author, unknown_author):
        """Calculate similarity of two author profiles."""
        diff = 0
        for feature_kn, value in known_author.features.items():
            diff += abs(value - unknown_author.features.get(feature_kn, 0))
        for feature_unk, value in unknown_author.features.items():
            if feature_unk not in known_author.features:
                diff += abs(-value)
        return diff

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.destroy()

# add if not yet in most common list
# –number of commas
# –number of dots
# –number of exclamation marks
# –number of question marks
# –number of colons
# –number of semicolons