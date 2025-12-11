import re
import pyttsx3


class TextToSpeech:
    def __init__(self, rate: int = 150, volume: float = 0.9):
        self.engine = pyttsx3.init()
        
        # Set to English (America) voice
        voices = self.engine.getProperty('voices')
        if voices:
            # Look for English (America) voice - ID: gmw/en-us
            best_voice = None
            for voice in voices:
                # Check by ID first (most reliable)
                if voice.id == 'gmw/en-us':
                    best_voice = voice
                    break
                # Also check by name
                voice_name_lower = voice.name.lower()
                if 'english' in voice_name_lower and 'america' in voice_name_lower:
                    best_voice = voice
                    break
            
            # Set the voice if we found one
            if best_voice:
                self.engine.setProperty('voice', best_voice.id)
        
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self.text_buffer = ""
        self.speaking = False
        self.stop_flag = False
    
    def speak(self, text: str):
        if not text.strip() or self.stop_flag:
            return
        
        self.speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.speaking = False
    
    def speak_streaming(self, text_chunk: str):
        if self.stop_flag:
            return
        
        self.text_buffer += text_chunk
        
        # Find complete sentences (ending with . ! ? followed by space or end)
        sentences = re.split(r'([.!?]\s+|\.$|!$|\?$)', self.text_buffer)
        
        # Process complete sentences
        i = 0
        while i < len(sentences) - 1:
            sentence = sentences[i].strip()
            if sentence:
                # Speak the complete sentence
                self.speaking = True
                self.speak(sentence)
                self.speaking = False
            i += 2  # Skip the punctuation/space separator
        
        # Keep the last incomplete sentence in buffer
        if len(sentences) % 2 == 1:
            self.text_buffer = sentences[-1]
        else:
            self.text_buffer = ""
    
    def flush(self):
        if self.text_buffer.strip():
            self.speaking = True
            self.speak(self.text_buffer.strip())
            self.speaking = False
            self.text_buffer = ""
    
    def stop(self):
        self.stop_flag = True
        self.text_buffer = ""
        try:
            self.engine.stop()
        except:
            pass


if __name__ == "__main__":
    # Test the TTS
    tts = TextToSpeech()
    tts.speak("Hello! This is a test of the text to speech system.")
    
    # Test streaming
    print("Testing streaming mode...")
    tts_stream = TextToSpeech()
    chunks = ["Hello", " there!", " How", " are", " you", " doing", " today?", " This", " is", " a", " streaming", " test."]
    for chunk in chunks:
        tts_stream.speak_streaming(chunk)
    tts_stream.flush()