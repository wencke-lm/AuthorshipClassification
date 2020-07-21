import unittest

from tests.mtld_unittest import MtldTestCase
from tests.author_model_unittest import OtherFeatureExtractionFileTestCase
from tests.author_model_unittest import OtherFeatureExtractionDirectoryTestCase


def main(verbosity):
    project_suite = unittest.TestSuite()
    project_suite.addTest(unittest.makeSuite(MtldTestCase))
    project_suite.addTest(unittest.makeSuite(OtherFeatureExtractionFileTestCase))
    project_suite.addTest(unittest.makeSuite(OtherFeatureExtractionDirectoryTestCase))

    project_runner = unittest.TextTestRunner(verbosity=verbosity)
    project_runner.run(project_suite)
