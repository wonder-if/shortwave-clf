import numpy as np
import librosa
import random


class AudioTransforms:
    def __init__(
        self,
        noise_prob=0.75,
        time_shift_prob=0.75,
        volume_change_prob=0.75,
        speed_change_prob=0.15,
        pitch_shift_prob=0.15,
        gaussian_noise_prob=0.75,
    ):
        self.noise_prob = noise_prob
        self.time_shift_prob = time_shift_prob
        self.volume_change_prob = volume_change_prob
        self.speed_change_prob = speed_change_prob
        self.pitch_shift_prob = pitch_shift_prob
        self.gaussian_noise_prob = gaussian_noise_prob

    def add_background_noise(self, audio, noise_level=0.1):
        if random.random() < self.noise_prob:
            noise = np.random.randn(len(audio)) * noise_level
            audio = audio + noise
        return audio

    def time_shift(self, audio, max_shift=0.1):
        if random.random() < self.time_shift_prob:
            shift = int(random.uniform(-max_shift, max_shift) * len(audio))
            audio = np.roll(audio, shift)
        return audio

    def speed_change(self, audio, sr, min_speed=0.8, max_speed=1.2):
        if random.random() < self.speed_change_prob:
            speed_factor = random.uniform(min_speed, max_speed)
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        return audio

    def volume_change(self, audio, min_volume=0.8, max_volume=1.2):
        if random.random() < self.volume_change_prob:
            volume_factor = random.uniform(min_volume, max_volume)
            audio = audio * volume_factor
        return audio

    def pitch_shift(self, audio, sr, min_steps=-2, max_steps=2):
        if random.random() < self.pitch_shift_prob:
            n_steps = random.randint(min_steps, max_steps)
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        return audio

    def add_gaussian_noise(self, audio, noise_level=0.1):
        if random.random() < self.gaussian_noise_prob:
            noise = np.random.normal(0, noise_level, len(audio))
            audio = audio + noise
        return audio

    def random_crop(self, audio, crop_length=2048):
        start = random.randint(0, len(audio) - crop_length)
        audio = audio[start : start + crop_length]
        return audio

    def __call__(self, audio, sr=44100):
        audio = self.time_shift(audio)
        if sr is not None:
            audio = self.speed_change(audio, sr)
            audio = self.pitch_shift(audio, sr)
        audio = self.add_gaussian_noise(audio)
        # clip len to 2048
        audio = self.random_crop(audio, crop_length=2048)
        return audio
