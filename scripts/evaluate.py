# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 28/08/2020
# Python 3.7.3
# Windows 8
"""Calculate the accuracy over the test set."""

import logging
import os
import sys

from tqdm import tqdm

# in order to access module from sister directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from lib.author_ident import AuthorIdent


LOG = logging.getLogger(__name__)
LOG.setLevel("INFO")
LOG.addHandler(logging.StreamHandler())


def evaluate(catalog, filename, test_dir):
    """Evaluate the accuracy of a trained system.

    Accuracies for the whole system and single authors are given
    out to the commandline, while a csv-file is created that
    contains classified files together with their gold standard
    and the predicted class.

    Args:
        catalog(str): Path to a catalog file created
            by AuthorIdent.
        filename(str): Name of the csv-file.
        test_dir(str): Path to a directory (i.e. called 'test')
            that contains folders named like the classes the
            given classifier is trained for that contain
            files belonging to this class.
    """
    with open(filename, 'w', encoding='utf-8') as eval_file:
        eval_file.write("file_id\tgold\tprediction\n")
        classifer = AuthorIdent(catalog)
        correct = 0
        total = 0
        for author in tqdm(os.listdir(test_dir), leave=False):
            correct_author = 0
            total_author = 0
            for file in tqdm(os.listdir(os.path.join("corpus", "test", author)), leave=False):
                result = classifer.classify(os.path.join(test_dir, author, file))
                if result == author:
                    correct += 1
                    correct_author += 1
                total += 1
                total_author += 1
                eval_file.write(f"{file}\t{author}\t{result}\n")
            LOG.info("Accuracy for {}: {:.2%}".format(author, correct_author/total_author))
        LOG.info("Total Accuracy: {:.2%}".format(correct/total))


if __name__ == "__main__":
    if len(sys.argv) != 4:
        LOG.error("Wrong number of commandline arguments.\n")
        LOG.info("Synopsis:")
        LOG.info("$ python scripts\\evaluate.py CATALOG FILENAME TEST_DIRECTORY\n")
        LOG.info("CATALOG         Path to the csv-file containing lines of the\n"
                 "                form <author>\\t<pretrained model JSON-filepath>\n"
                 "                created by AuthorIdent.")
        LOG.info("FILENAME        Where to save the results.")
        LOG.info("TEST_DIRECTORY  Path to the 'test' folder created by splitting the data.")
    else:
        evaluate(*sys.argv[1:])
