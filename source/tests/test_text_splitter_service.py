import unittest
from services.text_splitter_service import TextSplitterService

class TestTextSplitterService(unittest.TestCase):

    def test_split_text(self):
        text = "This is a test. " * 50
        result = TextSplitterService.split_text(text)
        self.assertTrue(isinstance(result, list))
        self.assertGreater(len(result), 0)

    def test_split_text_empty(self):
        result = TextSplitterService.split_text("")
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()