# File name: add_background_noise.py
# Function: Add Gaussian noise to an audio file with a given noise level
# Authors: Craig Droke, Jacob Boyle, Kristian DelSignore, and Matthew Zmuda
# Date: 12/13/2024

import numpy as np
from scipy.io import wavfile

def add_background_noise(input_file, output_file, noise_level=0.00005):
    # Read the input .wav file
    sample_rate, audio_data = wavfile.read(input_file)
    
    # Ensure audio_data is float
    audio_data = audio_data.astype(float)

    # Generate Gaussian noise
    noise = np.random.normal(0, noise_level * 32767, audio_data.shape)  # Scale by 32767 for int16 range

    # Add noise to the original signal
    noisy_signal = audio_data + noise
    
    # Convert back to int16 for saving as wav
    noisy_signal_int = np.clip(noisy_signal, -32768, 32767).astype(np.int16)  # Clip to prevent overflow

    # Save the noisy signal as a new .wav file
    wavfile.write(output_file, sample_rate, noisy_signal_int)

    return noisy_signal, noise

if __name__ == "__main__":
    add_background_noise("output_audio_0.1_amplitude.wav", "noisy_small_clip-Ally1205.wav")