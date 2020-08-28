# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik
# 4. Semester

# 14/07/2020
# Python 3.7.3
# Windows 8
"""Implementation of lexical diversity measure mtld."""

import logging
from types import GeneratorType

from lib.errors import ScarceDataError


LOG = logging.getLogger(__name__)


def _ttr(types, tokens):
    """(= type-token ratio (Templin, 1957))"""
    return types/tokens


def _mtld(seq, ttr_threshold=0.72, reverse=False):
    seq_len = 0  # number of tokens in the whole sequence
    seg_count = 0  # segments with a ttr of lower 0.72
    seq_ttr = 1.0  # ttr of previous segment
    token_count = 0  # (not unique) words in the current segment
    types = set()  # different words in the current segment
    for word in {True: reversed, False: lambda s: s}[reverse](seq):
        seq_len += 1
        token_count += 1
        types.add(word)
        seq_ttr = _ttr(len(types), token_count)
        # start new segment if ttr drops below 0.72
        if seq_ttr <= ttr_threshold:
            seg_count += 1
            token_count = 0
            types.clear()
    # collect incomplete segment by approximating how close it was to being completed
    if seq_ttr > ttr_threshold:
        seg_count += ((1.0 - seq_ttr)/(1.0 - ttr_threshold))
    if seg_count == 0:  # if there was only a single occurence of every word
        raise ScarceDataError("Can't calculate MTLD score of a sequence with no repeating words.")
    if seq_len < 100:
        LOG.warning("MTLD scores for sequences shorter than 200 words are not reliable.")
    return seq_len/seg_count


def mtld(seq):
    """(= measure of textual lexical diversity (McCarthy, 2005))

    Text length measured in words is divided by a segment count.
    Segments are computed by starting from a word and calculating the
    TTR for growing sequences beginning from this word on.
    As soon as the sequence's TTR is lower than 0.72 a new segment
    is started.
    The two MTLD values of the sequence and the reversed sequence
    are computed and their mean is returned as the final MTLD value.

    Args:
        seq (list): Text as a list of tokens.

    Returns:
        int: MTLD score.
    """
    if isinstance(seq, list):
        return (_mtld(seq) + _mtld(seq, reverse=True))/2
    if isinstance(seq, GeneratorType):
        return _mtld(seq)  # getting the reverse of an iterator is computationally difficult
    raise ValueError("The Input should be a list or generator. "
                     "Try using split if your input was a string.")
