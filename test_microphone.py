#! python3.7

import os
import numpy as np
import speech_recognition as sr
from scipy import signal
from sys import platform
from datetime import datetime


# ============================================================================
# MICROPHONE SETTINGS - Tune these values as needed
# ============================================================================
# Energy threshold for detecting speech (lower = more sensitive)
ENERGY_THRESHOLD = 1000

# Whether to dynamically adjust energy threshold based on ambient noise
DYNAMIC_ENERGY_THRESHOLD = False

# Sample rate (Hz) - Linux typically uses 44100, others use 16000
if 'linux' in platform:
    SAMPLE_RATE = 44100
else:
    SAMPLE_RATE = 16000

# Duration of recording in seconds
RECORDING_DURATION = 5.0

# Output filename
OUTPUT_FILENAME = f"test_recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

# Adjust for ambient noise before recording (seconds to sample)
ADJUST_FOR_AMBIENT_NOISE = True
AMBIENT_NOISE_DURATION = 1.0

# Audio amplification factor (multiply volume by this amount)
AMPLIFICATION_FACTOR = 3.0
# ============================================================================


def record_audio(duration: float = RECORDING_DURATION, 
                 energy_threshold: int = ENERGY_THRESHOLD,
                 dynamic_energy_threshold: bool = DYNAMIC_ENERGY_THRESHOLD,
                 sample_rate: int = SAMPLE_RATE,
                 adjust_for_ambient_noise: bool = ADJUST_FOR_AMBIENT_NOISE,
                 ambient_noise_duration: float = AMBIENT_NOISE_DURATION) -> sr.AudioData:
    """
    Record audio from the microphone for a specified duration.
    
    Args:
        duration: Recording duration in seconds
        energy_threshold: Energy threshold for speech detection
        dynamic_energy_threshold: Whether to dynamically adjust threshold
        sample_rate: Audio sample rate in Hz
        adjust_for_ambient_noise: Whether to adjust for ambient noise first
        ambient_noise_duration: Duration to sample ambient noise
        
    Returns:
        AudioData object containing the recorded audio
    """
    # Initialize speech recognizer
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = dynamic_energy_threshold
    
    # Set up microphone
    source = sr.Microphone(sample_rate=sample_rate)
    
    print(f"Microphone settings:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Energy threshold: {energy_threshold}")
    print(f"  Dynamic energy threshold: {dynamic_energy_threshold}")
    print(f"  Recording duration: {duration} seconds")
    print()
    
    # Adjust for ambient noise if requested
    if adjust_for_ambient_noise:
        print(f"Adjusting for ambient noise (sampling for {ambient_noise_duration} seconds)...")
        with source:
            recorder.adjust_for_ambient_noise(source, duration=ambient_noise_duration)
        print(f"  Adjusted energy threshold: {recorder.energy_threshold}")
        print()
    
    # Record audio
    print(f"Recording for {duration} seconds...")
    print("(Speak now!)")
    
    with source:
        audio = recorder.record(source, duration=duration)
    
    print("Recording complete!")
    print()
    
    return audio


def amplify_audio(audio: sr.AudioData, factor: float = AMPLIFICATION_FACTOR) -> sr.AudioData:
    """
    Amplify audio by a given factor using numpy.
    
    Args:
        audio: AudioData object to amplify
        factor: Amplification factor (e.g., 2.0 doubles the volume)
        
    Returns:
        New AudioData object with amplified audio
    """
    # Get raw audio data as bytes
    raw_data = audio.get_raw_data()
    
    # Convert to numpy array (16-bit signed integers)
    audio_np = np.frombuffer(raw_data, dtype=np.int16)
    
    # Amplify by multiplying
    amplified_np = (audio_np * factor).astype(np.int16)
    
    # Clip to prevent overflow (int16 range: -32768 to 32767)
    amplified_np = np.clip(amplified_np, -32768, 32767)
    
    # Convert back to bytes
    amplified_bytes = amplified_np.tobytes()
    
    # Create new AudioData object with amplified audio
    amplified_audio = sr.AudioData(
        amplified_bytes,
        audio.sample_rate,
        audio.sample_width
    )
    
    return amplified_audio


def save_audio_to_file(audio: sr.AudioData, filename: str = OUTPUT_FILENAME):
    """
    Save AudioData to a WAV file.
    
    Args:
        audio: AudioData object to save
        filename: Output filename
    """
    # Get the raw audio data
    audio_data = audio.get_raw_data()
    
    # Get audio parameters
    sample_rate = audio.sample_rate
    sample_width = audio.sample_width
    
    print(f"Saving audio to: {filename}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Sample width: {sample_width} bytes")
    print(f"  Duration: {len(audio_data) / (sample_rate * sample_width):.2f} seconds")
    
    # Save to WAV file
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    
    print(f"Audio saved successfully!")
    print(f"  File size: {os.path.getsize(filename) / 1024:.2f} KB")


def main():
    """Main function to record and save audio."""
    print("=" * 60)
    print("Microphone Test - Recording Audio")
    print("=" * 60)
    print()
    
    # Record audio with current settings
    audio = record_audio(
        duration=RECORDING_DURATION,
        energy_threshold=ENERGY_THRESHOLD,
        dynamic_energy_threshold=DYNAMIC_ENERGY_THRESHOLD,
        sample_rate=SAMPLE_RATE,
        adjust_for_ambient_noise=ADJUST_FOR_AMBIENT_NOISE,
        ambient_noise_duration=AMBIENT_NOISE_DURATION
    )
    
    # Amplify audio
    print(f"Amplifying audio by {AMPLIFICATION_FACTOR}x...")
    audio = amplify_audio(audio, AMPLIFICATION_FACTOR)
    print("Amplification complete!")
    print()
    
    # Save to file
    save_audio_to_file(audio, OUTPUT_FILENAME)
    
    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

