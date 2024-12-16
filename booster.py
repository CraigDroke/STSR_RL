# File name: booster.py
# Function: Boost the amplitude of an audio file by a given factor (for comparison with our main.py outputs)
# Authors: Craig Droke, Jacob Boyle, Kristian DelSignore, and Matthew Zmuda
# Date: 12/13/2024

import numpy as np
from scipy.io import wavfile

def boost_amplitude(input_file, output_file, boost_factor=1.5):
    # Read the input .wav file
    sample_rate, audio_data = wavfile.read(input_file)

    # Convert to float for processing
    audio_float = audio_data.astype(float)

    # Boost the amplitude
    boosted_audio = audio_float * boost_factor

    # Clip to prevent overflow
    boosted_audio = np.clip(boosted_audio, -32768, 32767)

    # Convert back to int16
    boosted_audio_int = boosted_audio.astype(np.int16)

    # Write the boosted audio to a new file
    wavfile.write(output_file, sample_rate, boosted_audio_int)

    print(f"Boosted audio saved to {output_file}")

# Example usage
input_file = "noisy_small_clip-Ally1205.wav"
output_file = "new_out6_boosted.wav"
boost_factor = 100  # Adjust this value to increase or decrease the boost

boost_amplitude(input_file, output_file, boost_factor)