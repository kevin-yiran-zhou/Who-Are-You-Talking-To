from language_model import LanguageModel
from speech_to_text import Whisper
from text_to_speech import TextToSpeech
from time import sleep


class VoiceAgent:
    def __init__(self):
        print("="*50)
        print("Loading Whisper (speech-to-text)...")
        self.whisper = Whisper()
        print("="*50)
        print("Loading Language Model...")
        self.llm = LanguageModel(verbose=False)
        print("="*50)
        print("Loading Text-to-Speech...")
        self.tts = TextToSpeech()
        print("="*50)
    
    def run(self):
        """Run the voice agent in continuous mode."""
        print("\n🎤 Listening...")
        print("Press Ctrl+C to stop\n")
        
        self.whisper.start_listening()
        processed_phrases = set()
        last_transcription_length = 0
        
        try:
            while True:
                # PREVENT SELF-LISTENING
                # If the agent is talking, we assume any sound is its own voice.
                # We clear the audio buffer so it doesn't hear itself.
                if self.tts.is_speaking():
                    self.whisper.data_queue.queue.clear()
                    sleep(0.1)
                    continue
                
                text = self.whisper.process_audio()
                
                if text is not None:
                    # Check if a new complete phrase was added
                    current_transcription = self.whisper.get_transcription()
                    current_length = len(current_transcription)
                    
                    # If transcription length increased, a new phrase was completed
                    if current_length > last_transcription_length and current_length > 1:
                        # Get the newly completed phrase (the last one)
                        user_input = current_transcription[-1].strip()
                        
                        # Process if we haven't seen this phrase before and it's not empty
                        if user_input and user_input not in processed_phrases:
                            print(f"\n🗣️  You: {user_input}")
                            print("🤖 Assistant: ", end="", flush=True)
                            
                            # Get response from LLM with streaming (both print and speak)
                            def stream_callback(text_chunk: str):
                                print(text_chunk, end="", flush=True)
                                self.tts.speak_streaming(text_chunk)
                            
                            response = self.llm.generate(
                                user_input,
                                stream_callback=stream_callback
                            )
                            
                            # Flush any remaining text in TTS buffer
                            self.tts.flush()
                            print("\n")
                            
                            processed_phrases.add(user_input)
                    
                    last_transcription_length = current_length
                else:
                    sleep(0.25)
                    
        except KeyboardInterrupt:
            print("\n\nStopping voice agent...")
        finally:
            self.whisper.stop_listening()
            print("Goodbye!")


if __name__ == "__main__":
    agent = VoiceAgent()
    agent.run()