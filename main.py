# -*- coding: utf-8 -*-

# Wencke Liermann
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 19/07/2020
# Python 3.7.3
# Windows 8
"""."""

import argparse
import logging

from scripts.author_ident import AuthorIdent


def main():
    parser = argparse.ArgumentParser(
        description="Manages author profiles and performs authorship attribution.",
        epilog="Training(source) files need to be preprocessed, "
               "containing one sentence per line and a space between tokens.") # usage= to adapt synopsis
    parser.add_argument('--catalog', nargs=1, metavar="filename",
                        help="Path to a file containing lines of the form "
                             r"<author>\t<pretrained model .csv-filename> .")
    parser.add_argument('--train', nargs=2, metavar=("author", "source"),
                        help="Add one new class to the classifier.")
    parser.add_argument('--forget', nargs=1, metavar="author",
                        help="Delete one class from the classifier.")
    parser.add_argument('--classify', nargs=1, metavar="source",
                        help="Returns the most likely author for the given text.")
    parser.add_argument('--test', help="Run all unittests.", action="store_true")
    parser.add_argument('--verbosity', type=int, choices=[0, 1, 2], default=1, 
                        help="Adjust the amount of output (0=errors, "
                             "1=warnings and above, 2=info and above).")

    args = parser.parse_args()
    logging.basicConfig(level=["ERROR", "WARNING", "INFO"][args.verbosity],
            format="%(name)s:%(levelname)s:%(message)s")
    if args.catalog:
        classifier = AuthorIdent(*args.catalog)
    if args.train:
        if not args.catalog:
            parser.error("--train requires --catalog.")
        try:
            classifier.train(*args.train)
        except AuthorError as e:
            logging.error(str(e))
    if args.forget:
        if not args.catalog:
            parser.error("--forget requires --catalog.")
        try:
            classifier.forget(*args.forget)
        except AuthorError as e:
            logging.error(str(e))
    if args.classify:
        if not args.catalog:
            parser.error("--classify requires --catalog.")
        try:
            classifier.classify(*args.classify)
        except AuthorError as e:
            logging.error(str(e))
    if args.test:
        import tests
        tests.main(args.verbosity)

if __name__ == "__main__":
    main()
