import unittest

from tests.mtld_unittest import MtldTestCase


def main(verbosity):
    project_suite = unittest.TestSuite()
    project_suite.addTest(unittest.makeSuite(MtldTestCase))

    project_runner = unittest.TextTestRunner(verbosity=verbosity)
    project_runner.run(project_suite)
