#! python3.7

import os
import numpy as np
import speech_recognition as sr
import whisper
import torch
from scipy import signal

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from typing import Optional, List, Callable


class Whisper:
    def __init__(self):
        """Initialize the Whisper speech-to-text system with default settings."""
        # Initialize speech recognizer
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 1000
        self.recorder.dynamic_energy_threshold = False
        
        # Set up microphone
        if 'linux' in platform:
            self.source = sr.Microphone(sample_rate=44100)
        else:
            self.source = sr.Microphone(sample_rate=16000)
        
        # Load Whisper model (base.en for English)
        self.audio_model = whisper.load_model("base.en")
        
        # Initialize state
        self.phrase_time = None
        self.data_queue = Queue()
        self.phrase_bytes = bytes()
        self.transcription = ['']
        self.is_listening = False
        self.stop_listener = None
        # record_timeout: Maximum length of each recording chunk in seconds
        # Increased from 2.0 to 30.0 to allow longer continuous speech/questions
        # The system will still detect phrase completion via silence (phrase_timeout)
        self.record_timeout = 30.0
        # phrase_timeout: Time of silence (in seconds) before considering phrase complete
        # If no new audio chunks arrive for this duration, the phrase is marked complete
        self.phrase_timeout = 3.0
        
        # Audio amplification factor (only used on Linux)
        self.amplification_factor = 6.0 if 'linux' in platform else 1.0
        
        # Adjust for ambient noise
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
    
    def _record_callback(self, _, audio: sr.AudioData) -> None:
        """Threaded callback function to receive audio data when recordings finish."""
        data = audio.get_raw_data()
        self.data_queue.put(data)
    
    def _amplify_audio_bytes(self, audio_bytes: bytes, factor: float) -> bytes:
        """Amplify audio bytes by a given factor using numpy.
        
        Args:
            audio_bytes: Raw audio bytes (16-bit PCM)
            factor: Amplification factor (e.g., 3.0 triples the volume)
            
        Returns:
            Amplified audio bytes
        """
        # Convert to numpy array (16-bit signed integers)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Amplify by multiplying
        amplified_np = (audio_np * factor).astype(np.int16)
        
        # Clip to prevent overflow (int16 range: -32768 to 32767)
        amplified_np = np.clip(amplified_np, -32768, 32767)
        
        # Convert back to bytes
        return amplified_np.tobytes()
    
    def start_listening(self):
        """Start listening to the microphone in the background."""
        if self.is_listening:
            return
        
        self.stop_listener = self.recorder.listen_in_background(
            self.source, 
            self._record_callback, 
            phrase_time_limit=self.record_timeout
        )
        self.is_listening = True
    
    def stop_listening(self):
        """Stop listening to the microphone."""
        if not self.is_listening:
            return
        
        if self.stop_listener:
            self.stop_listener(wait_for_stop=True)
        self.is_listening = False
    
    def process_audio(self) -> Optional[str]:
        """Process audio from the queue and return transcribed text if available.
        
        The phrase is considered "complete" when there's been silence (no new audio chunks)
        for phrase_timeout seconds. This allows for natural pauses in speech while still
        detecting when the user has finished speaking.
        
        Returns:
            Transcribed text if new audio was processed, None otherwise
        """
        if self.data_queue.empty():
            return None
        
        now = datetime.utcnow()
        phrase_complete = False
        
        # Check if phrase is complete: if more than phrase_timeout seconds have passed
        # since the last audio chunk, the user has likely finished speaking
        if self.phrase_time and now - self.phrase_time > timedelta(seconds=self.phrase_timeout):
            self.phrase_bytes = bytes()
            phrase_complete = True
        
        # Update the timestamp of the last audio chunk received
        self.phrase_time = now
        
        # Combine audio data from queue
        audio_data = b''.join(self.data_queue.queue)
        self.data_queue.queue.clear()
        
        # Add new audio data to accumulated phrase
        self.phrase_bytes += audio_data
        
        # Amplify audio on Linux
        if 'linux' in platform and self.amplification_factor > 1.0:
            self.phrase_bytes = self._amplify_audio_bytes(self.phrase_bytes, self.amplification_factor)
        
        # Convert to numpy array
        audio_np = np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Resample if needed (Linux uses 44100 Hz, Whisper expects 16000 Hz)
        if 'linux' in platform:
            original_sample_rate = 44100
            target_sample_rate = 16000
            num_samples = int(len(audio_np) * target_sample_rate / original_sample_rate)
            audio_np = signal.resample(audio_np, num_samples)
        
        # Transcribe
        result = self.audio_model.transcribe(audio_np, fp16=torch.cuda.is_available())
        text = result['text'].strip()
        
        # Update transcription
        if phrase_complete:
            self.transcription.append(text)
        else:
            self.transcription[-1] = text
        
        return text
    
    def get_transcription(self) -> List[str]:
        """Get the current transcription history."""
        return self.transcription.copy()
    
    def clear_transcription(self):
        """Clear the transcription history."""
        self.transcription = ['']
        self.phrase_bytes = bytes()
        self.phrase_time = None
    
    def run_continuous(self, callback: Optional[Callable[[str], None]] = None):
        """Run continuous transcription with real-time updates.
        
        Args:
            callback: Optional callback function called with each transcribed text
        """
        self.start_listening()
        
        try:
            while True:
                text = self.process_audio()
                if text is not None:
                    if callback:
                        callback(text)
                    else:
                        # Default behavior: clear console and print transcription
                        os.system('cls' if os.name == 'nt' else 'clear')
                        for line in self.transcription:
                            print(line)
                        print('', end='', flush=True)
                else:
                    sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_listening()


def main():
    """Command-line interface for the Whisper class."""
    whisper_stt = Whisper()
    whisper_stt.run_continuous()


if __name__ == "__main__":
    main()