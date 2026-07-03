import numpy as np

class Transcriber:
    def __init__(self, model_size='base'):
        self.model_size = model_size

    def transcribe(self, audio_segment):
        """
        Simulate speech-to-text. In production, use Whisper.
        Returns transcription and confidence.
        """
        # Simulate: random text or None
        if np.random.random() < 0.8:
            text = "This is a simulated transcription."
            confidence = np.random.uniform(0.8, 0.99)
        else:
            text = None
            confidence = 0.0
        return text, confidence