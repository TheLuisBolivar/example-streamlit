class TokenService:
    @staticmethod
    def estimate_tokens(text):
        """Estimate the number of tokens (1 token ≈ 4 characters)."""
        return len(text) / 4