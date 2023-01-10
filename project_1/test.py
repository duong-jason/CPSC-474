#!/usr/bin/env python3

import unittest
from project_1 import extract_vars, check_dependencies

class TestDependencies(unittest.TestCase):
    def test_parse(self):
        a = ["a = ( a + b ) * c", "a = a - a / b"]
        self.assertEqual(
            extract_vars(a), [[{"a"}, {"a", "b", "c"}], [{"a"}, {"a", "b"}]]
        )

    def test_false(self):
        a, b, = [
            "a = b"
        ], ["c = a", "b = c", "a = c"]
        for _ in b:
            with self.subTest(i=_):
                self.assertFalse(
                    check_dependencies(*extract_vars(a), *extract_vars([_]))
                )

    def test_true(self):
        a, b, = [
            "a = c"
        ], ["b = c"]
        self.assertTrue(check_dependencies(*extract_vars(a), *extract_vars(b)))

if __name__ == "__main__":
    unittest.main()
