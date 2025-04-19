import numpy as np


def add_noise_keypoints(squence_kpt, noise_level=0.01):
    if np.random.rand() > 0.5:
        return squence_kpt
    noise = np.random.uniform(-noise_level, noise_level, squence_kpt.shape)
    noisy_keypoints = squence_kpt + noise
    return noisy_keypoints

def interpolate_temporal(sequence_kpt, num_input_sequences=50, num_output_sequences=50):
    if np.random.rand() > 0.5:
        return sequence_kpt
    output_sequence = np.zeros((num_output_sequences, 17, 2))

    step = (num_input_sequences - 1) / (num_output_sequences - 1)
    output_sequence[0] = sequence_kpt[0]
    
    for i in range(1, num_output_sequences):
        current_step = step * i
        low_idx = int(np.floor(current_step))
        high_idx = min(int(np.ceil(current_step)), num_input_sequences - 1)

        alpha = current_step - low_idx

        kpt_low = sequence_kpt[low_idx]
        kpt_high = sequence_kpt[high_idx]

        kpt_interpolated = (1 - alpha) * kpt_low + alpha * kpt_high
        output_sequence[i] = kpt_interpolated

    return output_sequence