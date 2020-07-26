# -*- coding: utf-8 -*-

# Wencke Liermann
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 25/07/2020
# Python 3.7.3
# Windows 8
"""Representation of author profiles as feature vectors."""

from collections import Counter
import csv
import logging
import os

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

from scripts.mtld import mtld


LOG = logging.getLogger(__name__)  # module logger
MOST_FREQUENT_WORDS = os.path.join("data", "most_common_words.csv")


class AuthorModel:
    """Representation of a sample of an author's collected works.
    
    This representation as a numeric feature vector can be
    understood as an author profile.

    Attributes:
        features(dict): A mapping from feature names(str) to
            values(int) - feature vector.
    """
    def __init__(self):
        self.features = dict()

    @classmethod
    def train(cls, source):
        """Create new author profile.

        Args:
            source(str): Path to an utf8-encoded .txt-file.
                Alternatively one can also pass a directory containing
                such files.
                All the files must have already been preprocessed,
                separating tokens with whitespaces and giving each
                sentence a single line. One can use the function
                <AuthorModel.preprocess> to perform this preprocessing.
                

        Returns:
            AuthorModel: Newly created author profile.
        """
        if os.path.isfile(source):
            files = [source]
        elif os.path.isdir(source):
            files = [os.path.join(source, fl) for fl in os.listdir(source)]
            if files == []:
                LOG.error(f"Passed argument '{source}' matches an empty directory.")
                return None
        else:
            LOG.error(f"Passed argument '{source}' matches no file or directory.")
            return None

        author_pr = AuthorModel()
        author_pr._extract_features(files)
        return author_pr

    @classmethod
    def read_csv(cls, filename):
        """Load pretrained author profile.
        
        Args:
            filename(str): File created by the
                function <AuthorModel.write_csv>.

        Returns:
            AuthorModel: Loaded author profile.
        """
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"Passed argument '{filename}' matches no file.")

        author_pr = AuthorModel()
        with open(filename, 'r', encoding='utf-8') as file_in:
            csv_reader = csv.DictReader(file_in, delimiter='\t')
            for ln, line in enumerate(csv_reader, 1):
                try:
                    value = float(line["value"])
                except TypeError:
                    LOG.warning(f"Ignored line {ln}; "
                        "missing float value at second position.")
                else:
                    author_pr.features[line["feature"]] = value
        return author_pr

    def write_csv(self, filename):
        """Save pretrained model in tab-separated .csv-file.
        
        Lines have the form <feature_name>\t<normalized_frequency>.

        Args:
            filename (str): The model is saved under this name.

        Returns:
            None
        """
        with open(filename, 'w', encoding='utf-8') as file_out:
            file_out.write("feature\tvalue\n")
            for ftr, freq in self.features.items():
                file_out.write(f"{ftr}\t{freq}\n")

    @staticmethod
    def preprocess(source, target):
        """Convert files to the format required by <AuthorModel.train>.

        By the means of sentence tokenization each sentence is placed
        on its own line. The sentences are word tokenized and tokens
        separated by one space.

        Args:
            source(str): utf8-encoded .txt-file one wants to preprocess.
            target(str): Where to save the preprocessed version to.

        Returns:
            None
        """
        with open(source, 'r', encoding='utf-8') as file_in, \
             open(target, 'w', encoding='utf-8') as file_out:
            text = ''
            for line in file_in:
                text += line.rstrip() + ' '
            sents = tokenize.sent_tokenize(text)
            for s in sents:
                tokens = tokenize.word_tokenize(s)
                file_out.write(' '.join(tokens) + '\n')

######################################## private methods ########################################

    def _extract_features(self, files):
        """Fill feature vector with properties extracted from .txt-files.
        + average word length
        + average sentence length
        + sentence length distribution
        + pos-tag trigrams frequency
        + most common word frequency
        + mtld score on word level
        """
        cw = {"<none>":0}  # common word count; including function words
        with open(MOST_FREQUENT_WORDS, 'r', encoding='utf-8') as cw_file:
            for line in cw_file:
                cw[line.rstrip()] = 0
        trigram_dist = Counter()  # frequency counter of POS-tag trigrams
        sen_len_dist = Counter()  # frequency of certain sentence length ranges
        word_count, acc_word_len = 0, 0  # accumulated word length
        sen_count, acc_sen_len = 0, 0  # accumulated sentence length
        acc_mtld = 0  # accumulated mtld scores of all files

        for filename in tqdm(files):
            acc_mtld += mtld(self._get_words(filename))
            collect = []  # contains up to one complete trigram
            # features on sentence level
            for s in self._get_tagged_sentences(filename):
                sen_count += 1
                acc_sen_len += len(s)
                sen_len_dist[self._get_partition(len(s))] += 1
                # features on token level
                for token, tag in s:
                    lemma = self._get_lemma(token, tag)
                    if lemma in cw:
                        cw[lemma] += 1
                    else:
                        cw["<none>"] += 1
                    word_count += 1
                    acc_word_len += len(token)
                    collect.append(tag)
                    if len(collect) == 3:
                        trigram_dist[str(collect)] += 1
                        del collect[0]
        # word_count and sen_count can't be empty otherwise mtld would have thrown an error
        # normalize all features and add them to the feature vector
        self._add_feature(trigram_dist, normalize=((word_count - 2*len(files))))
        self._add_feature(sen_len_dist, normalize=sen_count)
        self._add_feature(("<average_word_length>", acc_word_len), normalize=word_count*10)
        self._add_feature(("<average_sen_length>", acc_sen_len), normalize=sen_count*100)
        self._add_feature(("<mtld_score>", acc_mtld), normalize=len(files)*100)
        self._add_feature(cw, normalize=word_count)

    def _add_feature(self, name_value, normalize=1):
        """Normalize feature before adding it to the feature vector."""
        if isinstance(name_value, dict):
            for ftr, freq in name_value.items():
                self.features[ftr] = round(freq/normalize, 10)
        elif isinstance(name_value, tuple):
            ftr, freq = name_value
            self.features[ftr] = round(freq/normalize, 10)

    @staticmethod
    def _get_words(filename):
        """Generator for words of a file"""
        with open(filename, 'r', encoding='utf-8') as file_in:
            for line in file_in:
                for token in line.split():
                    yield token

    @staticmethod
    def _get_tagged_sentences(filename):
        """Generator with progressbar for tagged sentences of a file."""
        with tqdm(total=os.stat(filename).st_size, leave=False) as pbar:
            with open(filename, 'r', encoding='utf-8') as file_in:
                for line in file_in:
                    pbar.update(len(line.encode('utf-8')) + 1)
                    tagged = pos_tag(line.split())
                    yield tagged

    @staticmethod
    def _get_partition(s_len):
        """Get interval for sentence length distribution."""
        if s_len <= 5:
            return "<sen_len_0-5>"
        if s_len <= 20:
            return "<sen_len_5-20>"
        if s_len <= 40:
            return "<sen_len_20-40>"
        if s_len <= 65:
            return "<sen_len_40-65>"
        if s_len <= 95:
            return "<sen_len_65-95>"
        if s_len <= 130:
            return "<sen_len_95-130>"
        return "<sen_len_130<>"

    @staticmethod
    def _get_lemma(word, nltk_pos_tag, lemmatizer=WordNetLemmatizer()):
        """Lemmatize a word."""
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        pos_tag = tag_dict.get(nltk_pos_tag[0])
        if pos_tag is None:
            return word.lower()
        return lemmatizer.lemmatize(word.lower(), pos_tag)

    # necessary for the unittests to work
    def __eq__(self, other):
        if self.features == other.features:
            return True
        return False

