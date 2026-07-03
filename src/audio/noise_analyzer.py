import numpy as np

class NoiseAnalyzer:
    def analyze_audio(self, audio_signal, sample_rate=16000):
        """
        Compute audio features: RMS, spectral centroid, etc.
        Simulated: return average dB level.
        """
        # Simulate dB level between 40 and 80
        db = np.random.uniform(40, 80)
        return {'db': db, 'rms': db - 30, 'spectral_centroid': np.random.uniform(1000, 5000)}