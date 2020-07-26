import unittest

from tests.mtld_unittest import MtldTestCase
from tests.author_model_unittest import FeatureExtractionFileTestCase
from tests.author_model_unittest import FeatureExtractionDirectoryTestCase


def main(verbosity):
    project_suite = unittest.TestSuite()
    project_suite.addTest(unittest.makeSuite(MtldTestCase))
    project_suite.addTest(unittest.makeSuite(FeatureExtractionFileTestCase))
    project_suite.addTest(unittest.makeSuite(FeatureExtractionDirectoryTestCase))

    project_runner = unittest.TextTestRunner(verbosity=verbosity)
    project_runner.run(project_suite)
