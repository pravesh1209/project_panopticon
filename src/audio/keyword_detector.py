import numpy as np

class KeywordDetector:
    def __init__(self, keywords=['cheat', 'answer', 'copy']):
        self.keywords = keywords

    def detect_keywords(self, text):
        """
        Check if any suspicious keyword is present in transcription.
        """
        if not text:
            return []
        found = [kw for kw in self.keywords if kw.lower() in text.lower()]
        return found