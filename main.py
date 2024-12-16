# File name: main.py
# Function: Complete STSR and use a reinforcement learning agent to learn the optimal noise level for each segment.
# Authors: Craig Droke, Jacob Boyle, Kristian DelSignore, and Matthew Zmuda
# Date: 12/13/2024

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter

# Define the Stochastic Resonance RL Environment class (will be the input signal itself)
class StochasticResonanceEnvironment:
    def __init__(self, signal, noise_level_range, num_steps, sample_rate):
        self.signal = signal
        self.noise_level_range = noise_level_range
        self.num_steps = num_steps
        self.current_step = 0
        self.sample_rate = sample_rate

    def step(self, action):
        # Get the noise level based on the action taken
        noise_level = self.noise_level_range[action]

        # Calculate the number of samples per segment TODO - could potentially make some type of array outside of this function to save time
        segment_size = len(self.signal) // self.num_steps

        # Determine the start and end indices of the current segment
        start_index = self.current_step * segment_size
        end_index = (self.current_step + 1) * segment_size

        # Extract the current segment
        segment = self.signal[start_index:end_index]

        # Perform FFT on segment 
        # NOTE in the paper that this is technically the DTF but this function does it more efficiently
        fft_result = np.fft.fft(segment)
        magnitudes = np.abs(fft_result)
        phases = np.angle(fft_result) 

        # Define m
        m = len(segment)

        # Calculate the frequency of each bin
        bin_frequencies = np.arange(m) * self.sample_rate / m

        # 6. Correct the loudness of the noise accoriding to the bin frequencies
        corrected_magnitudes = self.apply_mel_correction(magnitudes, bin_frequencies)

        # Add noise and apply thresholding
        noisy_magnitudes = np.maximum(corrected_magnitudes + noise_level * np.random.randn(len(corrected_magnitudes)), 0)
        thresholded_magnitudes = self.tsr_threshold_function(noisy_magnitudes)
        # NOTE: Below is suppressed and can be uncommented for only STSR testing
        # thresholded_magnitudes = self.tsr_threshold_function(corrected_magnitudes)

        # Reconstruct the signal using inverse FFT
        thresholded_fft_result = thresholded_magnitudes * np.exp(1j * phases)
        reconstructed_segment = np.real(np.fft.ifft(thresholded_fft_result))

        # Calculate reward
        reward = self.get_reward(segment, reconstructed_segment)

        # TODO - we need to apply a low-pass filter to the reconstructed segment (future work)
        # Apply low-pass filter to the reconstructed segment
        # reconstructed_filtered_segment = apply_lowpass_filter(reconstructed_segment, cutoff=500, fs=self.sample_rate)
        reconstructed_filtered_segment = reconstructed_segment

        # Update the current step
        self.current_step += 1

        # Check if the episode is done
        done = self.current_step >= self.num_steps

        # Return the state, reward, and done flag
        return self._get_state(), reward, done, reconstructed_filtered_segment
    
    def frequency_to_mel(self, f):
        return 2595 * np.log10(1 + f / 700)

    def mel_to_frequency(self, m):
        return 700 * (10**(m / 2595) - 1)

    def equal_loudness_contour(self, frequency):
        return 3.64 * (frequency/1000)**-0.8 - 6.5 * np.exp(-0.6 * (frequency/1000 - 3.3)**2) + 10**-3 * (frequency/1000)**4

    # Function to reset the environment to the beginning (of the signal)
    def reset(self, sample_message=None):
        self.current_step = 0  # Ensure current_step is reset
        # If sample_message is None, reset to the default starting point of the signal
        if sample_message is None:
            self.current_state = self.signal[self.current_step:self.current_step + self.num_steps]
        else:
            self.current_state = sample_message
        return self._get_state()
    
    def apply_mel_correction(self, magnitudes, bin_frequencies,num_mel_points=128):
        # Ensure inputs are 1D
        if magnitudes.ndim > 1:
            magnitudes = magnitudes.mean(axis=1)  # Average across channels if multi-dimensional
        if bin_frequencies.ndim > 1:
            bin_frequencies = bin_frequencies.flatten()
        # Create mel scale points
        mel_points = np.linspace(
            0, np.log10(sample_rate / 2 + 1e-10), num=num_mel_points
        )  # Mel scale points from 0 to Nyquist frequency
        freq_points = 10 ** mel_points  # Convert mel points back to Hz scale

        # Interpolate magnitudes to mel scale
        magnitudes_mel = np.interp(freq_points, bin_frequencies[: len(magnitudes)], magnitudes[: len(magnitudes)])

        # Apply equal-loudness correction
        corrected_magnitudes_mel = magnitudes_mel * self.equal_loudness_contour(freq_points)

        # Interpolate back to linear frequency scale
        corrected_magnitudes = np.interp(
            bin_frequencies[: len(magnitudes)], freq_points, corrected_magnitudes_mel
        )

        # Mirror corrected magnitudes for negative frequencies
        corrected_magnitudes_full = np.concatenate([corrected_magnitudes, corrected_magnitudes[-2:0:-1]])

        return corrected_magnitudes_full

    #TODO - we need to fix this function (its not collecting for all magnitudes)
    # Define thresholding function
    def tsr_threshold_function(self, noisy_magnitudes, threshold_value=0.1):
        for mag in noisy_magnitudes:
            if mag > threshold_value:
                mag = mag
            else:
                mag = 0
        return mag

    def _get_state(self):
        return self.current_step
    
    # Reward calculation function (spectral distance based)
    def get_reward(self, original_segment, reconstructed_segment):
        
        #NOTE - We do not think that SNR is valuable here as a reward because it is always going 
        # to recognize the good noise as bad. There may be additional parts required for this
        # reward function however. This would require weighting.

        # Calculate Spectral Distortion (using spectral distance)
        original_fft = np.fft.fft(original_segment)
        reconstructed_fft = np.fft.fft(reconstructed_segment)
        spectral_distance = np.mean(np.abs(original_fft - reconstructed_fft))

        # Invert the spectral distance to get the reward (want to minimize distance)
        reward = -spectral_distance

        return reward

# Define the Q-Learning Agent class
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate, discount_factor, epsilon):
        self.q_table = np.zeros((num_states + 1, num_actions))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
    
    def get_action(self, state):
        state = max(0, min(state, len(self.q_table) - 1))  # Clamp state

        # Epsilon-greedy exploration-exploitation trade-off
        if np.random.rand() < self.epsilon:
            # Exploration: Choose a random action
            action = np.random.randint(self.q_table.shape[1])
        else:
            # Exploitation: Choose the action with the highest Q-value
            action = np.argmax(self.q_table[state])

        # Decay epsilon
        self.epsilon *= 0.9999

        return action

    def update(self, state, action, reward, next_state):
        # Clamp states to valid range
        state = max(0, min(state, self.q_table.shape[0] - 1))
        next_state = max(0, min(next_state, self.q_table.shape[0] - 1))

        # Q-table update logic
        td_target = reward + self.discount_factor * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

        return td_error
    
# Non-class functions

# TODO - adjust and use this function (future work)
# def apply_lowpass_filter(data, cutoff, fs, order=5):
#     # Design the filter
#     nyquist = 0.5 * fs
#     normal_cutoff = cutoff / nyquist
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)

#     # Apply the filter
#     filtered_signal = lfilter(b, a, data)

#     return filtered_signal

# Main function
if __name__ == "__main__":

    # Import the audio signal, define the number of steps, environment, and agent
    sample_rate, signal = wavfile.read('noisy_small_clip-Ally1205.wav')
    num_steps = 8000
    noise_level_range = [10, 100, 1000] #TODO - we need to adjust this range (future work)
    env = StochasticResonanceEnvironment(signal=signal, noise_level_range=noise_level_range, num_steps=num_steps, sample_rate=sample_rate)
    agent = QLearningAgent(num_states=env.num_steps, num_actions=len(env.noise_level_range), 
                        learning_rate=0.1, discount_factor=0.99, epsilon=0.99)

    ############# TRAINING #############

    # Define the number of episodes
    num_episodes = 10

    # Train the agent for the defined number of episodes
    for episode in range(num_episodes):
        state = env.reset(sample_message=signal) # Reset the environment with the original signal
        done = False # Variable for breaking out of the loop
        total_reward = 0 # Total reward for the episode
        
        # While there are more signal segments to process
        while not done:
            action = agent.get_action(state) # Select an action
            next_state, reward, done, reconstructed_segment = env.step(action) # Take a step in the environment 
            state = next_state # Update the state with the next one
            total_reward += reward # Update the total episode reward
        
        # Print the total reward for the episode to the terminal
        print(f"Episode {episode}, Total Reward: {total_reward}") #Every episode right now to see progress

    ############# TESTING #############

    # Reset the environment for testing
    state = env.reset()
    done = False
    total_reward = 0
    reconstructed_segments = []

    # Test the trained agent (1 episode)
    while not done:
        action = agent.get_action(state) # Select a policy exploited action
        next_state, reward, done, reconstructed_segment = env.step(action)  # Step in the environment
        state = next_state  # Update the state
        total_reward += reward  # Accumulate the reward
        reconstructed_segments.append(reconstructed_segment)

    # Print the total reward for the test episode
    print(f"Test Episode, Total Reward: {total_reward}")

    # number of segments of output signal
    num_segments = len(reconstructed_segments)

    # Reconstruct the output signal by concatenating the reconstructed segments
    output_signal = np.concatenate(reconstructed_segments)

    # Define time axis for both signals
    time_axis_output = np.arange(len(output_signal)) / sample_rate  # Time axis for output signal
    time_axis_original = np.arange(len(signal)) / sample_rate  # Time axis for original signal

    # Create subplots for original and noisy signals
    plt.figure(figsize=(15, 10))

    # Subplot for Original Signal
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    plt.plot(time_axis_original, signal[:, 0], label='Original Signal', linewidth=1.5)
    plt.title('Original Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Subplot for Noisy Signal
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
    plt.plot(time_axis_output, output_signal[:, 0], label='Noisy Signal', linestyle='--', color='orange', linewidth=1.5)
    plt.title('Noisy Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Calculate the FFT and frequency axis for both signals
    fft_original = np.fft.fft(signal[:, 0])
    fft_noisy = np.fft.fft(output_signal[:, 0])

    # Frequency axis based on output signal length
    freq_axis_output = np.fft.fftfreq(len(output_signal[:, 0]), d=1/sample_rate)

    # Only plot the positive frequencies for both signals
    positive_freq_mask_output = freq_axis_output >= 0

    # Create subplots for amplitude spectrum of original and noisy signals
    plt.figure(figsize=(15, 10))

    # Subplot for Original Signal Amplitude Spectrum
    plt.subplot(2, 1, 1)
    freq_axis_original = np.fft.fftfreq(len(signal[:, 0]), d=1/sample_rate)
    positive_freq_mask_original = freq_axis_original >= 0
    plt.plot(freq_axis_original[positive_freq_mask_original], np.abs(fft_original[positive_freq_mask_original]))
    plt.title('Amplitude Spectrum of Original Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Subplot for Noisy Signal Amplitude Spectrum
    plt.subplot(2, 1, 2)
    plt.plot(freq_axis_output[positive_freq_mask_output], np.abs(fft_noisy[positive_freq_mask_output]))
    plt.title('Amplitude Spectrum of Noisy Signal')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # Normalize the signal if it's in float format
    if output_signal.dtype == np.float32 or output_signal.dtype == np.float64:
        # Scale from [-1, 1] to [−32768, 32767] for int16
        output_signal = np.int16(output_signal / np.max(np.abs(output_signal)) * 32767)

    # Define the output file path
    output_file = 'new_out7.wav'

    # Save the noisy signal as a WAV file
    wavfile.write(output_file, sample_rate, output_signal[:, 0])

    # Save the Q-table for analysis
    np.save('q_table.npy', agent.q_table)

