# -*- coding: utf-8 -*-

# Wencke Liermann
# Universität Potsdam
# Bachelor Computerlinguistik

# 14/07/2020
# Python 3.7.3
# Windows 8
"""mtld.py unittest class."""

import unittest

from mtld import mtld


class MtldTestCase(unittest.TestCase):
    def test_sequence_with_no_word_occuring_twice(self):
        with self.assertRaises(ValueError, msg="Expected a ValueError to be raised."):
            mtld("there was a tree near my house .".split())

    def test_sequence_splitted_into_segments_with_no_rest(self):
        mtld_score = mtld("the people for the people".split())
        self.assertEqual(mtld_score, 5, msg=f"Expected a mtld score of 5. Got {mtld_score}.")

    def test_sequence_splitted_into_segments_with_rest(self):
        mtld_score = mtld("the boy and the other boy went to a garden to play".split())
        self.assertEqual(round(mtld_score, 4), 10.4812,
                         msg=f"Expected a mtld score of 10.4812. Got {mtld_score}.")

    def test_on_longer_text_with_several_segments(self):
        with open("let_it_go_frozen.txt", encoding='utf-8') as resource:
            mtld_score = mtld(resource.read().split())
            self.assertEqual(round(mtld_score, 4), 53.3137,
                             msg=f"Expected a mtld score of 53.3137. Got {mtld_score}.")


if __name__ == "__main__":
    unittest.main()
