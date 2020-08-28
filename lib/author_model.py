# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 27/08/2020
# Python 3.7.3
# Windows 8
"""Representation of author profiles as feature matrices."""

from collections import namedtuple
import json
import logging
import os

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm

from lib.distribution import Distribution, IntegerDistribution
from lib.errors import ScarceDataError, log_exception
from lib.mtld import mtld


LOG = logging.getLogger(__name__)  # module logger
FREQ_WRDS = os.path.join("data", "most_common_words.csv")


class AuthorModel:
    """Represents author profiles as features matrices.

    This feature matrices are made up of 6 feature vectors
    of different lengths and with features belonging to
    different categories. They are calculated
    from a sample of an author's collected works and can
    be understood as his or her prototypical writing style.

    Attributes:
        word_len_distr(distribution.IntegerDistribution)
        sent_len_distr(distribution.IntegerDistribution)
        pos_trigram_distr(distribution.Distribution)
        freq_word_distr(distribution.Distribution):
            The 90 most common lemmatized tokens
            over all the tokens in the training set.
        punctuation_distr(distribution.Distribution)
        mtld(float): Lexical Diversity Score.
    """
    def __init__(self):
        self.word_len_distr = IntegerDistribution()
        self.sent_len_distr = IntegerDistribution()
        self.pos_trigram_distr = Distribution()
        self.freq_word_distr = Distribution()
        self.punctuation_distr = Distribution()
        self.mtld = 0

    @classmethod
    @log_exception(LOG)
    def train(cls, source):
        """Calculate a feature matrix from text samples.

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
            files = [os.path.join(source, file) for file in os.listdir(source)
                     if os.path.isfile(os.path.join(source, file))]
            if files == []:
                raise FileNotFoundError("Method 'train' requires the directory to contain files.")
        else:
            LOG.info(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Passed argument '{source}' matches no file or directory.")
        profile = AuthorModel()
        profile._extract_features(files)
        return profile

    @classmethod
    @log_exception(LOG)
    def read_json(cls, source):
        """Load precalculated feature matrix from file.

        Args:
            source(str): File created by the function
                <AuthorModel.write_json> containing
                one array consisting of five objects
                and one number.

        Returns:
            AuthorModel: Loaded author profile.
        """
        if not os.path.isfile(source):
            LOG.info(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Passed argument '{source}' matches no file.")

        profile = AuthorModel()
        with open(source, 'r', encoding='utf-8') as file_in:
            data = json.load(file_in, object_hook=cls._objectkeys_to_ints)
            if isinstance(data, list) and len(data) == 6:
                profile.word_len_distr = IntegerDistribution(data[0])
                profile.sent_len_distr = IntegerDistribution(data[1])
                profile.pos_trigram_distr = Distribution(data[2])
                profile.freq_word_distr = Distribution(data[3])
                profile.punctuation_distr = Distribution(data[4])
                if isinstance(data[5], (float, int)):
                    profile.mtld = data[5]
                else:
                    raise TypeError("The loaded array has to contain a number at last position.")
            else:
                raise ValueError("Wrong format; make sure to load a JSON-array of length six.")
        return profile

    def write_json(self, goal):
        """Save instance attributes in a json-file.

        Saved as an array of five objects and one number
        following the order used in <AuthorModel.__init__>.

        Args:
            goal(str): Location/name for the file.
        """
        with open(goal, 'w', encoding='utf-8') as file_out:
            content = [self.word_len_distr, self.sent_len_distr,
                       self.pos_trigram_distr, self.freq_word_distr,
                       self.punctuation_distr, self.mtld]
            json.dump(content, file_out, indent=4, default=lambda x: getattr(x, 'distr'))

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
                 open(goal, 'w', encoding='utf-8') as file_out:
                text = ''  # not ideal saving the text as whole
                for line in file_in:
                    text += line.rstrip() + ' '
                sents = sent_tokenize(text)
                for s in sents:
                    tokens = word_tokenize(s)
                    file_out.write(' '.join(tokens) + '\n')
        elif os.path.isdir(source):
            if not os.path.isdir(goal):
                os.makedirs(goal)
            for file in os.listdir(source):
                if os.path.isfile(os.path.join(source, file)):
                    cls.preprocess(os.path.join(source, file), os.path.join(goal, file))
        else:
            LOG.info(f"Current working directory: {os.getcwd()}")
            raise FileNotFoundError(f"Passed argument '{source}' matches no file or directory.")

    @log_exception(LOG)
    def normalized_feature_vector(self):
        """Normalized class attributes with additional statistics.

        Additional statistics include the mean and standard
        deviation of word and sentence length.
        For word length the feature encoding the
        relative frequency of words of length e.g. 5
        is renamed <w5>.
        For sentence length the feature encoding the
        relative frequency of sentences of length e.g. 5
        is renamed <s5>.

        Returns:
            dict: Unfolded feature matrix to a feature vector.
        """
        # word_len_distr, freq_word_distr are checked implicitely via pos_trigram_distr
        if (self.pos_trigram_distr.total < 2
                or self.word_len_distr.total < 1
                or self.punctuation_distr.total < 1):
            raise ScarceDataError("Not enough input data.\n"
                                  "At least two sentences, three consecutive words and\n"
                                  "one punctuation mark ('.', ';', ',', '?', '!')\n"
                                  "have to be included in the data.")
        vector = dict()
        vector.update({f"<w{key}>": value for key, value
                       in self.word_len_distr.prob_dist().items()})
        vector["<mean_word_len>"] = self.word_len_distr.mean()
        vector["<stdev_word_len>"] = self.word_len_distr.stdev(m=vector["<mean_word_len>"])
        vector.update({f"<s{key}>": value for key, value
                       in self.sent_len_distr.prob_dist().items()})
        vector["<mean_sent_len>"] = self.sent_len_distr.mean()
        vector["<stdev_sent_len>"] = self.sent_len_distr.stdev(m=vector["<mean_sent_len>"])
        vector.update(self.pos_trigram_distr.prob_dist())
        vector.update(self.freq_word_distr.prob_dist())
        vector.update(self.punctuation_distr.prob_dist())
        vector["<mtld_score>"] = self.mtld
        return vector

#################
# private methods
#################

    @log_exception(LOG)
    def _extract_features(self, files):
        """Build up feature vectors."""
        for file in tqdm(files):
            try:
                self.mtld += mtld(self._get_words(file))
            except ScarceDataError as exc:
                raise ScarceDataError(
                    f"File '{file}' inappropriate for feature extraction.") from exc
            collect = []  # collects up to three pos tags
            for sent in self._nlp(file):
                self.sent_len_distr[len(sent)] += 1
                for word in sent:
                    if word.punct:
                        self.punctuation_distr[word.text] += 1
                    else:
                        self.word_len_distr[len(word.text)] += 1
                    if word.freq_wrd:
                        self.freq_word_distr[word.lemma] += 1
                    else:
                        self.freq_word_distr["<none>"] += 1
                    collect.append(word.tag)
                    if len(collect) == 3:
                        self.pos_trigram_distr[str(collect)] += 1
                        del collect[0]
        self.mtld /= len(files)

    @staticmethod
    def _get_words(filename):
        """Generate words of a file."""
        with open(filename, 'r', encoding='utf-8') as file_in:
            for line in file_in:
                for token in line.split():
                    yield token

    @staticmethod
    def _nlp(filename):
        """
        Yields the sentences of a file one at a time and
        as a sequence of tokens with additional information.
        """
        # read in frequent words of interest
        with open(FREQ_WRDS, 'r', encoding='utf-8') as file_in:
            freq_wrd_lst = [wrd.rstrip() for wrd in file_in]
        # create struct to represent a single item
        Item = namedtuple('Item', ['text', 'lemma', 'tag', 'freq_wrd', 'punct'])

        # emulation of the spacy nlp pipeline
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
                        if token in {'.', ';', ',', '?', '!'}:
                            punct = True
                        items.append(Item(token, lemma, tag, freq_wrd, punct))
                    yield items

    @staticmethod
    def _get_lemma(word, nltk_pos_tag, lemmatizer=WordNetLemmatizer()):
        """Lemmatize a word."""
        # as the lemmatizer works on the wordnet tag set while the
        # default pos tagger follows the nltk tag set, first a translation
        # from nltk tags to wordnet tags has to take place
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        pos = tag_dict.get(nltk_pos_tag[0])
        if pos is None:
            return word.lower()
        return lemmatizer.lemmatize(word.lower(), pos)

    @staticmethod
    def _objectkeys_to_ints(obj):
        """Turn JSON-object keys that are numeric to integers."""
        return {(int(key) if key.isnumeric() else key): value for key, value in obj.items()}

    def __eq__(self, other):
        if isinstance(other, AuthorModel):
            return self.__dict__ == other.__dict__
        return False
