import unittest
from services.token_service import TokenService

class TestTokenService(unittest.TestCase):

    def test_estimate_tokens(self):
        text = "This is a test"
        result = TokenService.estimate_tokens(text)
        self.assertEqual(result, len(text) / 4)

    def test_estimate_tokens_empty(self):
        result = TokenService.estimate_tokens("")
        self.assertEqual(result, 0)

if __name__ == '__main__':
    unittest.main()