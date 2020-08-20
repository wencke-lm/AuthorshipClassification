# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 19/08/2020
# Python 3.7.3
# Windows 8
"""Representation of author profiles as feature vectors."""

from collections import namedtuple
import logging
import os

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

from scripts.distribution import Distribution, IntegerDistribution
from scripts.mtld import mtld
from scripts.errors import ScarceDataError, log_exception


LOG = logging.getLogger(__name__)  # module logger
FREQ_WRDS = os.path.join("data", "most_common_words.csv")


class AuthorModel(dict):
    """Dict subclass to represent a numeric feature vector.

    This feature vector calculated from a sample of an
    author's collected works can be understood as an
    author profile.

    The following features are examined:
        + mean/standard deviation for word length
        + mean/standard deviation for sentence length
        + distribution of sentence lengths
        + distribution of word lengths
        + relative frequencies of POS-tag trigrams
        + relative frequencies of common tokens
        + mtld score on token level
    """

    @classmethod
    @log_exception(LOG)
    def train(cls, source):
        """Calculate a feature vector from text samples.

        Args:
            source(str): Path to an utf8-encoded txt-file.
                Alternatively one can also pass a directory
                containing such files.
                All the files must have already been
                preprocessed, separating tokens with
                whitespaces and giving each sentence
                a single line. One can use the function
                <AuthorModel.preprocess> to perform
                this preprocessing.

        Returns:
            AuthorModel: New author profile.
        """
        if os.path.isfile(source):
            files = [source]
        elif os.path.isdir(source):
            files = [os.path.join(source, fl) for fl in os.listdir(source)
                     if os.path.isfile(os.path.join(source, fl))]
            if files == []:
                raise FileNotFoundError(f"method 'train' requires the directory to contain files")
        else:
            LOG.info(f"current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"passed argument '{source}' matches no file or directory")
        profile = AuthorModel()
        profile._extract_features(files)
        return profile

    @classmethod
    @log_exception(LOG)
    def read_csv(cls, source):
        """Load precalculated feature vector from file.

        Args:
            source(str): File created by the function
                <AuthorModel.write_csv> with lines of
                the format: <feature_name>\t<numeric_value>.

        Returns:
            AuthorModel: Loaded author profile.
        """
        if not os.path.isfile(source):
            LOG.info(f"current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"passed argument '{source}' matches no file")

        profile = AuthorModel()
        with open(source, 'r', encoding='utf-8') as file_in:
            for ln, line in enumerate(file_in, 1):
                line = line.rstrip().split('\t')
                if len(line) == 2:
                    try:
                        value = float(line[1])
                    except ValueError:
                        LOG.warning(f"ignored line {ln}; not-float value in second column")
                        LOG.info(f"lines should have the format: <feature_name>\t<numeric_value>")
                    else:
                        profile[line[0]] = value
                else:
                    LOG.warning(f"ignored line {ln}; missing column")
                    LOG.info(f"lines should have the format: <feature_name>\t<numeric_value>")
        return profile

    def write_csv(self, goal):
        """Save calculated feature vector in tab-separated file.

        Lines have the format <feature_name>\t<numeric_value>.

        Args:
            goal(str): Location/name for the file.
        """
        with open(goal, 'w', encoding='utf-8', newline="") as file_out:
            for ftr, freq in self.items():
                file_out.write(f"{ftr}\t{freq}\n")

    @classmethod
    @log_exception(LOG)
    def preprocess(cls, source, goal):
        """Convertion to the format required by <AuthorModel.train>.

        By the means of sentence tokenization each sentence is
        placed on its own line. The sentences are word tokenized
        and tokens separated by a single space.

        Args:
            source(str): Path to an utf8-encoded txt-file.
                Alternatively one can also pass a directory
                containing such files.
            goal(str): Location/name for the preprocessed version.
        """
        LOG.info(f"Preprocessing '{source}'...")
        if os.path.isfile(source):
            with open(source, 'r', encoding='utf-8') as file_in, \
                 open(goal, 'w', encoding='utf-8', newline="") as file_out:
                text = ''
                for line in file_in:
                    text += line.rstrip() + ' '
                sents = sent_tokenize(text)
                for s in sents:
                    tokens = word_tokenize(s)
                    file_out.write(' '.join(tokens) + '\n')
        elif os.path.isdir(source):
            if not os.path.isdir(goal):
                os.mkdir(goal)
            for fl in os.listdir(source):
                if os.path.isfile(os.path.join(source, fl)):
                    cls.preprocess(os.path.join(source, fl), os.path.join(goal, fl))
        else:
            LOG.info(f"current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"passed argument '{source}' matches no file or directory")

#################
# private methods
#################

    @log_exception(LOG)
    def _extract_features(self, files):
        """Build up feature vector"""
        word_len_dist = IntegerDistribution()
        sent_len_dist = IntegerDistribution()
        trigram_dist = Distribution()  # POS-tag trigrams
        freq_word_dist = Distribution()  # 90 most common lemmas in trainings set
        punctuation_dist = Distribution()
        acc_mtld = 0

        for fl in tqdm(files):
            try:
                acc_mtld += mtld(self._get_words(fl))
            except ScarceDataError as e:
                raise ScarceDataError(f"file '{fl}' not appropriate for training") from e
            collect = []
            for sent in self._nlp(fl):
                sent_len_dist.inc(len(sent))
                for word in sent:
                    if word.punct:
                        punctuation_dist.inc(word.text)
                    else:
                        word_len_dist.inc(len(word.text))
                    if word.freq_wrd:
                        freq_word_dist.inc(word.lemma)
                    else:
                        freq_word_dist.inc("<none>")
                    collect.append(word.tag)
                    if len(collect) == 3:
                        trigram_dist.inc(str(collect))
                        del collect[0]

        # word_len_dist, freq_word_dist are checked implicitely via trigram_dist
        if (sent_len_dist.total() < 2 or trigram_dist.total() < 1 or punctuation_dist.total() < 1):
            LOG.info(f"at least two sentences, three consecutive words and one punctuation mark "
                     "have to be included in the data")
            raise ScarceDataError(f"not enough input data")

        self.update({f"<w{key}>": value for key, value in word_len_dist.prob_dist().items()})
        self["<mean_word_len>"] = word_len_dist.mean()
        self["<stdev_word_len>"] = word_len_dist.stdev(m=self["<mean_word_len>"])
        self.update({f"<s{key}>": value for key, value in sent_len_dist.prob_dist().items()})
        freq_word_dist.plot("oh my",[])
        self["<mean_sent_len>"] = sent_len_dist.mean()
        self["<stdev_sent_len>"] = sent_len_dist.stdev(m=self["<mean_sent_len>"])
        self.update(trigram_dist.prob_dist())
        self.update(freq_word_dist.prob_dist())
        self.update(punctuation_dist.prob_dist())
        self["<mtld_score>"] = (acc_mtld/len(files))

    @staticmethod
    def _get_words(filename):
        """Generate words of a file."""
        with open(filename, 'r', encoding='utf-8') as file_in:
            for line in file_in:
                for token in line.split():
                    yield token

    @staticmethod
    def _nlp(filename):
        """Yields the sentences of a file one at a time and
           as a sequence of tokens with additional information.
        """
        # read in frequent words of interest
        with open(FREQ_WRDS, 'r', encoding='utf-8') as file_in:
            freq_wrd_lst = [wrd.rstrip() for wrd in file_in]
        # create struct to represent a single item
        Item = namedtuple('Item', ['text', 'lemma', 'tag', 'freq_wrd', 'punct'])

        with tqdm(total=os.stat(filename).st_size, leave=False) as pbar:
            with open(filename, 'r', encoding='utf-8') as file_in:
                for line in file_in:
                    pbar.update(len(line.encode('utf-8')) + 1)
                    items = []
                    for token, tag in pos_tag(line.split()):
                        lemma = AuthorModel._get_lemma(token, tag)
                        freq_wrd = False
                        if lemma in freq_wrd_lst:
                            freq_wrd = True
                        punct = False
                        if token in ['.', ';', ',', '?', '!']:
                            punct = True
                        items.append(Item(token, lemma, tag, freq_wrd, punct))
                    yield items

    @staticmethod
    def _get_lemma(word, nltk_pos_tag, lemmatizer=WordNetLemmatizer()):
        """Lemmatize a word."""
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        pos = tag_dict.get(nltk_pos_tag[0])
        if pos is None:
            return word.lower()
        return lemmatizer.lemmatize(word.lower(), pos)
