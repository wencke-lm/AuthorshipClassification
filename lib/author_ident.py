# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 19/08/2020
# Python 3.7.3
# Windows 8
"""Classifier for feature-based authorship attribution."""

import logging
import os

from lib.author_model import AuthorModel
from lib.errors import CatalogError, log_exception


LOG = logging.getLogger(__name__)


class AuthorIdent:
    """Collects author profiles in a catalog.

    As part of the project a catalog called 'gutenbergident.txt'
    containing pretrained models for the following 10 authors
    can be accessed:
        + Anthony Trollope
        + Charles Dickens
        + Charlotte Mary Yone
        + George Alfred Henty
        + Henry Rider Haggard
        + James Fenimore Cooper
        + R M Ballantyne
        + Robert Louis Stevenson
        + Sir Walter Scott
        + William Dean Howells

    The trainings data for this catalog has been taken from:
    Lahiri, S. (2014). Complexity of Word Collocation Networks:
    A Preliminary Structural Analysis. In Proceedings of the Student
    Research Workshop at the 14th Conference of the European Chapter
    of the Association for Computational Linguistics (pp. 96–105).
    Association for Computational Linguistics.

    Args:
        catalog(str): Path to a file containing lines of the form
            <author>\t<pretrained model csv-filename> .

    Attributes:
        catalog(str): Stored catalog filename.
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

    @log_exception(LOG)
    def train(self, author, source):
        """Add new author profile to classifier.

        Args:
            author(str): Name of the newly created class.
            source(str): Path to an utf8-encoded txt-file.
                Alternatively one can also pass a directory containing
                such files.
                All the files must have already been preprocessed,
                separating tokens with whitespaces and giving each
                sentence a single line. One can use the function
                <AuthorModel.preprocess> to perform this preprocessing.
        """
        if author in self.profiles:
            raise CatalogError(f"An entry for '{author}' already exists.")
        LOG.info(f"Add entry for '{author}'...")
        profile = AuthorModel.train(source)
        self.profiles[author] = profile
        filepath = os.path.join(os.path.dirname(self.catalog), author)
        if os.path.isfile(filepath + ".csv"):
            i = 2
            while os.path.isfile(filepath + f"({i}).csv"):
                i += 1
            filepath = filepath + f"({i})"
        profile.write_csv(filepath + ".csv")
        with open(self.catalog, 'a', encoding='utf-8') as file_out:
            file_out.write(f"{author}\t{filepath +'.csv'}\n")
        self.catalog_content[author] = filepath + ".csv"

    @log_exception(LOG)
    def forget(self, author):
        """Remove author profile from classifier.

        The saved pretrained model of this author will be deleted
        in the process as well.

        Args:
            author(str): Name of the class to be deleted.
        """
        if author not in self.catalog_content:
            raise CatalogError(f"No entry for '{author}' exists.")
        LOG.info(f"Delete entry for '{author}'...")
        self.profiles.pop(author)
        saved_model = self.catalog_content.pop(author)
        if os.path.isfile(saved_model):
            os.remove(saved_model)
        else:
            LOG.warning(f"The pretrained model '{saved_model}' "
                        "of the deleted author couldn't be removed.")
        with open(self.catalog, 'w', encoding='utf-8') as file_out:
            for author, profile in self.catalog_content.items():
                file_out.write(f"{author}\t{profile}\n")

    @log_exception(LOG)
    def classify(self, source):
        """Perform authorship attribution for given txt-file.

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
            raise CatalogError(f"'{self.catalog}' is trained for less than two authors.")
        LOG.info(f"Classify '{source}'...")
        unknown_author_pr = AuthorModel.train(source)
        best_match = (float("inf"), None)
        for known_author, known_author_pr in self.profiles.items():
            diff = self._simil(known_author_pr, unknown_author_pr)
            LOG.info(f"Difference score with '{known_author}': {diff}")
            if diff < best_match[0]:
                best_match = (diff, known_author)
        return best_match[1]

    @log_exception(LOG)
    def accuracy(self, input_vec):
        """Calculate the accuracy of an annotated test set.

        Args:
            input_vec(list<tuple>): Pairs containing the data to
                be classified at first position and its correct
                class at second position.

        Returns:
            float: Number of correctly annotated texts divided by the
                number of texts inputed.
        """
        if input_vec == []:
            raise ValueError("Accuracy of an empty test set can't be calculated.")
        correct = 0
        for data, sol in input_vec:
            if self.classify(data) == sol:
                correct += 1
        return correct/len(input_vec)

    def destroy(self):
        """Delete the catalog and all pretrained models linked to it."""
        LOG.info(f"Delete catalog '{self.catalog}'...")
        # type cast so a new object is created
        # the original object is modified while iterating
        for author in list(self.catalog_content.keys()):
            self.forget(author)
        if os.path.isfile(self.catalog):
            os.remove(self.catalog)
        else:
            LOG.warning(f"Catalog file '{self.catalog}' could not be deleted.")

#################
# private methods
#################

    def _read_catalog(self, catalog):
        """Load saved classifier."""
        if os.path.isfile(catalog):
            LOG.info(f"Load existing classifier with the catalog '{catalog}'...")
            with open(catalog, 'r', encoding='utf-8') as file_in:
                for ln, line in enumerate(file_in, 1):
                    line = line.rstrip().split('\t')
                    if len(line) == 2:
                        author, file = line
                        try:
                            profile = AuthorModel.read_csv(file)
                        except FileNotFoundError:
                            LOG.warning(f"Ignored line {ln}; could not open the file "
                                        f"'{file}' supposed to contain the pretrained model.")
                        else:
                            LOG.info(f"Trained for '{author}'.")
                            self.profiles[author] = profile
                            self.catalog_content[author] = file
                    else:
                        LOG.warning(f"Ignored line {ln}; missing column.")
                        LOG.info(f"Correct line format: <author_name>\t<saved_profile_file> .")
        else:
            answer = None
            while answer not in ['y', 'n']:
                answer = input("Catalog not found. Do you want to create the catalog? y/n\n")
                if answer == 'y':
                    LOG.info(f"Create new classifier with the catalog '{catalog}'...")
                    with open(catalog, 'w', encoding='utf-8'):
                        pass
                elif answer == 'n':
                    return

    @staticmethod
    def _simil(known_author, unknown_author):
        """Calculate similarity of two author profiles."""
        weights = {"<mean_word_len>": 0.05, "<stdev_word_len>": 0.05,
                   "<mean_sent_len>": 0.005, "<stdev_sent_len>": 0.005,
                   "<mtld_score>": 0.01}
        diff = 0
        for feature_kn, value in known_author.items():
            diff += abs(value-unknown_author.get(feature_kn, 0)) * weights.get(feature_kn, 1)
        for feature_unk, value in unknown_author.items():
            if feature_unk not in known_author:
                diff += abs(-value)*weights.get(feature_unk, 1)
        return diff
