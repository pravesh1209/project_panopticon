import pytest
from src.audio.transcriber import Transcriber

def test_transcriber():
    t = Transcriber()
    text, conf = t.transcribe(None)
    assert conf >= 0