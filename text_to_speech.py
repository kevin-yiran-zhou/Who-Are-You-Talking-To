import threading
import queue
import re
import time
import subprocess
import os
import tempfile
import platform
from pathlib import Path

class TextToSpeech:
    def __init__(self, voice_model: str = "en_US-lessac-medium", rate: float = 1.0, volume: float = 1.0):
        self.queue = queue.Queue()
        self.voice_model = voice_model
        self.rate = rate
        self.volume = volume
        self.stop_flag = False
        
        # Track speaking status
        self._is_speaking = False
        self._speaking_lock = threading.Lock()
        self._last_speak_time = 0
        self._cooldown_period = 0.5  # Wait 0.5 seconds after speaking stops before listening
        
        # Find voice model file
        self.model_path = self._find_voice_model()
        if self.model_path is None:
            raise FileNotFoundError(
                f"Voice model '{voice_model}' not found. "
                f"Please download it with: python3 -m piper.download_voices {voice_model}"
            )
        
        # This thread will handle the speaking in the background
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

        self.text_buffer = ""
    
    def _find_voice_model(self) -> Path:
        # Check common locations for the .onnx file
        possible_paths = [
            Path(f"{self.voice_model}.onnx"),
            Path(".") / f"{self.voice_model}.onnx",
            Path.home() / ".local" / "share" / "piper" / "voices" / f"{self.voice_model}.onnx",
            Path.cwd() / f"{self.voice_model}.onnx",
        ]
        
        for path in possible_paths:
            if path.exists() and path.suffix == '.onnx':
                return path
        
        return None

    def _worker(self):
        while True:
            # Wait for text to arrive in the queue
            text = self.queue.get()
            if text is None: break # Sentinel to stop
            
            try:
                if not self.stop_flag and text.strip():
                    # Mark as speaking
                    with self._speaking_lock:
                        self._is_speaking = True
                    
                    # Generate audio using Piper
                    self._speak_with_piper(text)
                    
            except Exception as e:
                print(f"TTS Error: {e}")
            finally:
                # Mark as not speaking and record the time
                with self._speaking_lock:
                    self._is_speaking = False
                    self._last_speak_time = time.time()
                self.queue.task_done()
    
    def _speak_with_piper(self, text: str):
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            wav_path = tmp_file.name
        
        try:
            # Generate audio using Piper
            # Piper reads text from stdin and writes WAV to output file
            cmd = [
                'python3', '-m', 'piper',
                '--model', str(self.model_path),
                '--output_file', wav_path
            ]
            
            # Add length scale (rate) if supported/needed. 
            # Note: Piper CLI args might differ slightly based on version for rate control.
            # Usually --length-scale (inverse of rate). 
            # If rate=1.0 (normal), scale=1.0. If rate=2.0 (fast), scale=0.5.
            if self.rate != 1.0:
                 cmd.extend(['--length_scale', str(1.0/self.rate)])

            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(input=text.strip())
            
            if process.returncode != 0:
                raise RuntimeError(f"Piper TTS failed: {stderr}")
            
            # Check if WAV file was created
            if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                raise RuntimeError("Piper TTS did not generate audio file")
            
            # Play the audio file
            if platform.system() == 'Linux':
                # -q for quiet mode
                subprocess.run(['aplay', '-q', wav_path], check=True)
            elif platform.system() == 'Darwin':
                subprocess.run(['afplay', wav_path], check=True)
            else:  # Windows
                # Use powershell to play sound to avoid opening a player window
                subprocess.run(['powershell', '-c', f'(New-Object Media.SoundPlayer "{wav_path}").PlaySync()'], check=True)
                
        finally:
            # Clean up temporary file
            try:
                if os.path.exists(wav_path):
                    os.unlink(wav_path)
            except:
                pass

    def speak(self, text: str):
        self.queue.put(text)

    def speak_streaming(self, text_chunk: str):
        if self.stop_flag: return
        
        self.text_buffer += text_chunk
        
        # Split by sentence endings (. ! ?)
        sentences = re.split(r'([.!?]\s+|\.$|!$|\?$)', self.text_buffer)
        
        i = 0
        while i < len(sentences) - 1:
            sentence = sentences[i].strip()
            if sentence:
                self.speak(sentence) # This is now instant/non-blocking
            i += 2
            
        if len(sentences) % 2 == 1:
            self.text_buffer = sentences[-1]
        else:
            self.text_buffer = ""

    def flush(self):
        if self.text_buffer.strip():
            self.speak(self.text_buffer.strip())
            self.text_buffer = ""
    
    def is_speaking(self):
        with self._speaking_lock:
            # Check if currently speaking (engine running)
            if self._is_speaking:
                return True
            
            # Check if queue has pending items
            if not self.queue.empty():
                return True
            
            # Check cooldown period (audio might still be echoing)
            if time.time() - self._last_speak_time < self._cooldown_period:
                return True
            
            return False

    def stop(self):
        self.stop_flag = True
        # Clear the queue
        with self.queue.mutex:
            self.queue.queue.clear()