import torchaudio
import torch

def compute_feats(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
    waveform = waveform.to(torch.float)
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000)(waveform)
    waveform = waveform.to(self.device)
    feats = kaldi.fbank(waveform,
                num_mel_bins=80,
                frame_length=25,
                frame_shift=10,
                energy_floor=0.0,
                sample_frequency=16000)
    feats = feats.unsqueeze(0)
    return feats

def cut_and_merge_audio(audio_file, num_parts=3, min_part_duration=0.05):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_file, normalize=False)

    # Calculate duration and part duration
    total_duration = waveform.size(1) / sample_rate
    part_duration = total_duration / num_parts
    print(f"total duration: {total_duration}")
    print(f"part duration: {part_duration}")
    # Cut and merge audio into parts
    audio_parts = []
    current_part = waveform[:, :0]  # Empty tensor to start with
    for i in range(num_parts):
        start_time = int(i * part_duration * sample_rate)
        end_time = int((i + 1) * part_duration * sample_rate)

        # Ensure end_time is within the audio length
        end_time = min(end_time, waveform.size(1))

        # Extract part
        part = waveform[:, start_time:end_time]

        # Check if the part duration is less than the minimum
        if (end_time - start_time) / sample_rate < min_part_duration:
            # Merge with the previous part
            current_part = torch.cat([current_part, part], dim=1)
        else:
            # Save the previous part and reset current_part
            audio_parts.append(current_part)
            current_part = part

    # Add the last part
    audio_parts.append(current_part)

    return audio_parts


def cut_audio_fixed_duration(audio_file, duration=1.0):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
    print(f"total waveform size {waveform.size()}")
    total_duration = waveform.size(1) / sample_rate
    print(f"total duration: {total_duration}")

    # Calculate the number of parts based on the desired duration
    num_parts = int(waveform.size(1) / (duration * sample_rate))

    # Cut audio into fixed-duration parts
    audio_parts = []
    for i in range(num_parts):
        start_time = int(i * duration * sample_rate)
        end_time = int((i + 1) * duration * sample_rate)

        # Ensure end_time is within the audio length
        end_time = min(end_time, waveform.size(1))

        # Extract part
        part = waveform[:, start_time:end_time]

        audio_parts.append(part)

    # Check if there is remaining audio after the last chunk
    last_chunk = int((num_parts) * duration * sample_rate)
    print(last_chunk)
    remaining_audio = waveform[:, last_chunk:]
    if remaining_audio.size(1) > 0:
        # Concatenate the remaining audio with the last chunk
        audio_parts[-1] = torch.cat([audio_parts[-1], remaining_audio], dim=1)

    return audio_parts

# Example usage:
# audio_file = "path/to/your/audio/file.wav"
# duration = 1.0  # Duration of each audio part in seconds
# parts = cut_audio_fixed_duration(audio_file, duration)
# Example usage:
audio_file = "/vinbrain/chuongct98/data/vlsp2021/VLSP_Zalo_Speech_Corpus/zalo/program-0041/program-0041-00420.wav"
num_parts = 3
min_part_duration = 0.05
parts = cut_audio_fixed_duration(audio_file)

# Now, `parts` is a list containing the waveform for each part of the audio.
# You can do further processing or analysis with these parts.
for part in parts:
    print(part.size())