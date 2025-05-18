import unittest
from string_utils import StringManipulator

class TestStringManipulator(unittest.TestCase):
    
    def setUp(self):
        self.manipulator = StringManipulator()
    
    def test_reverse_string(self):
        self.assertEqual(self.manipulator.reverse_string("hello"), "olleh")
        self.assertEqual(self.manipulator.reverse_string(""), "")
        self.assertEqual(self.manipulator.reverse_string("radar"), "radar")
    
    def test_count_characters(self):
        self.assertEqual(self.manipulator.count_characters("hello world"), 11)
        self.assertEqual(self.manipulator.count_characters(""), 0)
        self.assertEqual(self.manipulator.count_characters("a"), 1)
    
    def test_error_handling(self):
        with self.assertRaises(TypeError):
            self.manipulator.reverse_string(123)
        
        with self.assertRaises(TypeError):
            self.manipulator.count_characters(123)

if __name__ == "__main__":
    unittest.main()
