# -*- coding: utf-8 -*-
import unittest
import pep8

import os.path


class TestCodeStyle(unittest.TestCase):

    def test_pep8_conformance(self):
        tests_dir = os.path.dirname(__file__)
        modules_files = os.path.abspath(os.path.join(tests_dir, "..", "raytracer"))
        pep8style = pep8.StyleGuide(max_line_length=200, ignore=['E731'])
        result = pep8style.check_files([modules_files])
        self.assertEqual(result.total_errors, 0, "Found pep8 conformance issues")

if __name__ == '__main__':
    unittest.main()
