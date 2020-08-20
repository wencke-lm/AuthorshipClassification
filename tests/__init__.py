# -*- coding: utf-8 -*-

# Wencke Liermann - wliermann@uni-potsdam.de
# Universität Potsdam
# Bachelor Computerlinguistik

# 20/08/2020
# Python 3.7.3
# Windows 8
"""Collection of all project testcases."""

import unittest

from tests.mtld_unittest import *
from tests.author_model_unittest import *
from tests.author_ident_unittest import *


def main(verbosity):
    project_suite = unittest.TestSuite()
    project_suite.addTest(unittest.makeSuite(AccuracyTestCase))
    project_suite.addTest(unittest.makeSuite(ClassifyTestCase))
    project_suite.addTest(unittest.makeSuite(FeatureExtractionDirectoryTestCase))
    project_suite.addTest(unittest.makeSuite(FeatureExtractionFileTestCase))
    project_suite.addTest(unittest.makeSuite(ForgetTestCase))
    project_suite.addTest(unittest.makeSuite(InitTestCase))
    project_suite.addTest(unittest.makeSuite(IOInteractionTestCase))
    project_suite.addTest(unittest.makeSuite(MtldTestCase))
    project_suite.addTest(unittest.makeSuite(TrainTestCase))

    project_runner = unittest.TextTestRunner(verbosity=verbosity)
    project_runner.run(project_suite)
