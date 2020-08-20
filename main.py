# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 19/07/2020
# Python 3.7.3
# Windows 8
"""."""

import argparse
import json
import logging
import logging.config
import os

from scripts.author_ident import AuthorIdent
from scripts.author_model import AuthorModel


LOG = logging.getLogger(__name__)


class LoggingErrorFilter(logging.Filter):
    """Prevent raised errors to be logged to the console handler."""
    def filter(self, record):
        return record.levelno != logging.ERROR

def main():
    # configure commandline parser
    parser = argparse.ArgumentParser(
        description="Manages author profiles and performs authorship attribution.",
        epilog="Source files need to be preprocessed, "
               "containing one sentence per line and a space between tokens.")
    parser.add_argument('--catalog', nargs=1, metavar="filename",
                        help="Path to a file containing lines of the form "
                             r"<author>\t<pretrained model csv-filename> .")
    parser.add_argument('--classify', nargs=1, metavar="source",
                        help="Return the most likely author for the given text.")
    parser.add_argument("--destroy", action="store_true",
                        help="Delete a catalog and its content.")
    parser.add_argument('--forget', nargs=1, metavar="author",
                        help="Delete class from classifier.")
    parser.add_argument('--preprocess', nargs=1, metavar="filename",
                        help="Preprocess a raw txt-file.")
    parser.add_argument('--test', help="Run all unittests.", action="store_true")
    parser.add_argument('--train', nargs=2, metavar=("author", "source"),
                        help="Add new class to classifier.")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=1, 
                        help="Adjust the amount of output (0=errors, 1=warnings and above,"
                             " 2=info and above). Default is 1.")

    # parse commandline input
    args = parser.parse_args()

    # configure logging
    with open(os.path.join("data", "LogConfigDict.json"), "r") as fd:
        setting = json.load(fd)
        setting["filters"]["errorfilter"]["()"] = LoggingErrorFilter
        setting["handlers"]["console"]["level"] = ["ERROR", "WARNING", "INFO"][args.verbosity]
        logging.config.dictConfig(setting)
    
    # action targets
    if args.catalog:
        classifier = AuthorIdent(*args.catalog)
    if args.classify:
        if not args.catalog:
            parser.error("--classify requires --catalog.")
        else:
            result = classifier.classify(*args.classify)
            LOG.info(f"{args.classify[0]} classified as '{result}'.")
            if args.verbosity < 2:
                print(result)
    if args.destroy:
        if not args.catalog:
            parser.error("--destroy requires --catalog.")
        else:
            classifier.destroy()
    if args.forget:
        if not args.catalog:
            parser.error("--forget requires --catalog.")
        else:
            classifier.forget(*args.forget)
    if args.preprocess:
        AuthorModel.preprocess(*args.preprocess)
    if args.train:
        if not args.catalog:
            parser.error("--train requires --catalog.")
        else:
            classifier.train(*args.train)

    # test target
    if args.test:
        import tests
        tests.main(args.verbosity)

if __name__ == "__main__":
    main()
