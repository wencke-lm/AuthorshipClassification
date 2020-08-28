# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik

# 26/07/2020
# Python 3.7.3
# Windows 8
"""mtld.py testcase."""

import logging
import os
import unittest

from lib.errors import ScarceDataError
from lib.mtld import mtld, LOG


LOG.setLevel(logging.CRITICAL)


class MtldTestCase(unittest.TestCase):
    def test_on_generator(self):
        def get_words():
            filename = os.path.join("tests", "data", "frozen", "let_it_go_frozen.txt")
            with open(filename, encoding='utf-8') as resource:
                for word in resource.read().split():
                    yield word
        self.assertAlmostEqual(mtld(get_words()), 53.9259, places=4,
                               msg="Calculated score(left) different from expected(right).")

    def test_on_longer_text_with_several_segments(self):
        filename = os.path.join("tests", "data", "frozen", "let_it_go_frozen.txt")
        with open(filename, encoding='utf-8') as resource:
            self.assertAlmostEqual(mtld(resource.read().split()), 49.2634, places=4,
                                   msg="Calculated score(left) different from expected(right).")

    def test_sequence_splitted_into_segments_with_no_rest(self):
        mtld_score = mtld("the people for the people".split())
        self.assertEqual(mtld_score, 5,
                         msg="Calculated score(left) different from expected(right).")

    def test_sequence_splitted_into_segments_with_rest(self):
        mtld_score = mtld("the boy and the other boy went to a garden to play".split())
        self.assertAlmostEqual(mtld_score, 10.4812, places=4,
                               msg="Calculated score(left) different from expected(right).")

    def test_sequence_with_no_word_occuring_twice(self):
        with self.assertRaises(ScarceDataError):
            mtld("there was a tree near my house .".split())
