# -*- coding: utf-8 -*-

# Wencke Liermann
# Universität Potsdam
# Bachelor Computerlinguistik

# 21/07/2020
# Python 3.7.3
# Windows 8
"""Model for feature based authorship attribution."""

from collections import Counter
import logging
import os
import re

import nltk

from mtld import mtld

LOG = logging.getLogger(__name__)
MOST_FREQUENT_WORDS = os.path.join("data", "most_common_words.csv")


class AuthorModel:
    """Author profile.
    
    Representation of a sample of an author's collected works as
    a non-binary feature vector.

    Attributes:
        features (dict): A mapping from feature names (str) to
            values (int) - feature vector.
    """
    def __init__(self):
        self.features = dict()

    def train(self, source):
        """Create new author profile(model).

        Args:
            source (str): Path to an utf8-encoded .txt file
                or directory with such files. All the files
                must have already been preprocessed, separating
                single tokens with whitespaces and containing one
                sentence per line. The contents of
                these shall be used to train the model on.

        Returns:
            None
        """
        if os.path.isfile(source):
            files = [source]
        elif os.path.isdir(source):
            files = [os.path.join(source, fl) for fl in os.listdir(source)]
            if files == []:
                LOG.error(f"Passed argument {source} matches an empty directory.")
                return None
        else:
            LOG.error(f"Passed argument {source} matches no file or directory.")
            return None
        self._extract_other_features(files)
        self._extract_syntactic_features()

    def read_csv(self, filename):
        """Load pretrained author profile(model)
        
        Args:
            filename (str): File created by the function write_csv.

        Returns:
            None
        """
        with open(filename, 'r', encoding='utf-8') as file_in:
            for ln, line in enumerate(file_in, 1):
                content = line.split('\t')
                if len(content) == 2:
                    try:
                        value = int(content[1])
                    except ValueError:
                        LOG.warning(f"Ignored line {ln}, "
                            "missing integer value at 2. position.")
                    else:
                        self.features[content[0]] = value
                else:
                    LOG.warning(f"Ignored line {ln}, "
                        "too many tap-separated columns.")

    def write_csv(self, filename):
        """Save pretrained model in tap-separated .csv file.

        Args:
            filename (str)

        Returns:
            None
        """
        with open(filename, 'w', encoding='utf-8') as file_out:
            for ftr in self.features:
                file_out.write(f"{ftr}\t{self.features[ftr]}\n")

    def _extract_other_features(self, files):
        """
        - average word length
        - average sentence length
        - sentence length distribution
        - POS-Tag trigram distribution
        - mtld score on word level
        - most common word frequency
        """
        cw = dict()  # counts common words frequencies including function words
        with open(MOST_FREQUENT_WORDS, 'r', encoding='utf-8') as cw_file:
            for line in cw_file:
                cw[line.rstrip()] = 0
        trigram_dist = Counter()  # frequency counter of POS-tag trigrams
        sen_len_dist = Counter()  # frequency of certain sentence length ranges
        word_count, acc_word_len = 0, 0  # accumulated word length
        sen_count, acc_sen_len = 0, 0  # accumulated sentence length
        acc_mtld = 0  # accumuated mtld scores of all files

        for filename in files:
            acc_mtld += mtld(self._get_words(filename))
            collect = []  # contains up to one complete trigram
            # features on sentence level
            for s in self._get_tagged_sentences(filename):
                sen_count += 1
                acc_sen_len += len(s)
                sen_len_dist[self._get_partition(len(s))] += 1
                # features on token level
                for token, tag in s:
                    if token.lower() in cw:
                        cw[token.lower()] += 1
                    word_count += 1
                    acc_word_len += len(token)
                    collect.append(tag)
                    if len(collect) == 3:
                        trigram_dist[str(collect)] += 1
                        del collect[0]
        # word_count and sen_count can't be empty otherwise mtld would have thrown an error
        # normalize all features and add them to the feature vector
        self._add_feature(trigram_dist, normalize=((word_count - 2*len(files))/100))
        self._add_feature(sen_len_dist, normalize=sen_count)
        self._add_feature(("average_word_length", acc_word_len), normalize=word_count)
        self._add_feature(("average_sen_length", acc_sen_len), normalize=sen_count)
        self._add_feature(("mtld_score", acc_mtld), normalize=len(files))
        self._add_feature(cw, normalize=word_count/100)

    def _extract_syntactic_features(self):
        pass

    def _add_feature(self, name_value, normalize=1):
        """Normalize feature before adding it to the feature vector."""
        if isinstance(name_value, dict):
            for name in name_value:
                self.features[name] = round(name_value[name]/normalize, 6)
        elif isinstance(name_value, tuple):
            name, value = name_value
            self.features[name] = round(value/normalize, 6)

    @staticmethod
    def _get_partition(x):
        """Get interval for sentence length distribution."""
        if x <= 5:
            return "sen_len_0-5"
        if x <= 20:
            return "sen_len_5-20"
        if x <= 40:
            return "sen_len_20-40"
        if x <= 65:
            return "sen_len_40-65"
        if x <= 95:
            return "sen_len_65-95"
        if x <= 130:
            return "sen_len_95-130"
        return "sen_len_130<"

    @staticmethod
    def _get_tagged_sentences(filename):
        """Generator - tagged sentences of a file"""
        with open(filename, 'r', encoding='utf-8') as file_in:
            for line in file_in:
                tokens = line.split()
                tagged = nltk.pos_tag(tokens)
                yield tagged

    @staticmethod
    def _get_words(filename):
        """Generator - words of a file"""
        with open(filename, 'r', encoding='utf-8') as file_in:
            for line in file_in:
                tokens = line.split()
                for token in tokens:
                    yield token


if __name__ == "__name__":
    a = AuthorModel()
    a.train("frozen")
    print(a.features)