# -*- coding: utf-8 -*-

# Wencke Liermann
# Universität Potsdam
# Bachelor Computerlinguistik

# 19/07/2020
# Python 3.7.3
# Windows 8
"""."""

import argparse
import logging


def main():
    parser = argparse.ArgumentParser(
        description="Manages author profiles and performs authorship attribution.")
    parser.add_argument('--test', help="Run all unittests.", action="store_true")
    parser.add_argument('--verbosity',
            help="Adjust the amount of output (0=errors, 1=warnings and above, 2=info and above).",
            type=int, choices=[0, 1, 2], default=1)

    args = parser.parse_args()
    logging.basicConfig(level=["ERROR", "WARNING", "INFO"][args.verbosity],
            format="\n%(name)s:%(levelname)s:%(message)s")
    if args.test:
        import tests
        tests.main(args.verbosity)


main()
