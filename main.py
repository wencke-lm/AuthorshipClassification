# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 28/08/2020
# Python 3.7.3
# Windows 8
"""Project script."""

import argparse
import json
import logging
import logging.config
import os
import sys

from lib.author_ident import AuthorIdent
from lib.author_model import AuthorModel


LOG = logging.getLogger(__name__)
ROOT = os.path.dirname(os.path.abspath(__file__))


class LoggingErrorFilter(logging.Filter):
    """Prevent raised errors to be logged to the console handler."""
    def filter(self, record):
        return record.levelno != logging.ERROR


def configure_parser():
    """Define targets and overall commandline layout."""
    parser = argparse.ArgumentParser(
        description="Manage author profiles and perform authorship attribution.",
        epilog="Source files need to be preprocessed, containing "
               "one sentence per line and a space between tokens.")
    parser.add_argument('--catalog', nargs=1, metavar="CATALOG",
                        help="Path to a file containing lines of the form "
                             r"<author>\t<pretrained model json-filename> .")
    parser.add_argument('--classify', nargs=1, metavar="SOURCE",
                        help="Return the most likely author for the given text.")
    parser.add_argument("--destroy", action="store_true",
                        help="Delete a catalog and its content.")
    parser.add_argument('--forget', nargs=1, metavar="AUTHOR",
                        help="Delete class from classifier.")
    parser.add_argument('--preprocess', nargs=2, metavar=("FILENAME", "GOAL"),
                        help="Preprocess a raw txt-file.")
    parser.add_argument('--test', help="Run all unittests.", action="store_true")
    parser.add_argument('--train', nargs=2, metavar=("AUTHOR", "SOURCE"),
                        help="Add new class to classifier.")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=1,
                        help="Adjust the amount of output (0=errors, 1=warnings "
                             "and above, 2=info and above). Default is 1.")
    return parser


def configure_logging(verbosity):
    """Create handlers and set levels for logging."""
    with open(os.path.join(ROOT, "data", "log_config.json"), "r", encoding='utf-8') as fd:
        setting = json.load(fd)
        setting["filters"]["errorfilter"]["()"] = LoggingErrorFilter
        setting["handlers"]["console"]["level"] = ["ERROR", "WARNING", "INFO"][verbosity]
        logging.config.dictConfig(setting)


def execute_commands(args):
    """Access the addressed methods from AuthorModel and AuthorIdent."""
    # test target
    if args.test:
        import tests
        tests.main(args.verbosity)

    # action targets
    if args.preprocess:
        AuthorModel.preprocess(*args.preprocess)
    if args.catalog:
        try:
            classifier = AuthorIdent(*args.catalog)
        except FileNotFoundError:
            answer = None
            while answer not in {'y', 'n'}:
                answer = input("Catalog not found. Do you want to create the catalog? y/n\n")
                if answer == 'y':
                    LOG.info(f"Create new classifier with the catalog '{args.catalog[0]}'...")
                    with open(*args.catalog, 'w', encoding='utf-8'):
                        pass
                elif answer == 'n':
                    return
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
    if args.train:
        if not args.catalog:
            parser.error("--train requires --catalog.")
        else:
            classifier.train(*args.train)


if __name__ == "__main__":
    parser = configure_parser()
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        configure_logging(args.verbosity)
        execute_commands(args)
