import cv2
from language_model import LanguageModel
from speech_to_text import Whisper
from text_to_speech import TextToSpeech
from gaze_detector import GazeDetector

class VoiceAgent:
    def __init__(self):
        print("="*50)
        print("Loading Gaze Detector (Webcam)...")
        self.gaze = GazeDetector()
        print("Loading Whisper (speech-to-text)...")
        self.whisper = Whisper()
        print("Loading Language Model...")
        self.llm = LanguageModel(verbose=False)
        print("Loading Text-to-Speech...")
        self.tts = TextToSpeech()
        print("="*50)
    
    def run(self):
        print("\n🤖 ROBOT ONLINE")
        self.whisper.start_listening()
        processed_phrases = set()
        last_transcription_length = 0
        
        try:
            while True:
                # 1. SEE: Update video (Now keeps running while talking!)
                frame, is_looking = self.gaze.process_frame()
                if frame is not None:
                    status_color = (0, 255, 0) if is_looking else (0, 0, 255)
                    cv2.circle(frame, (30, 30), 15, status_color, -1)
                    
                    # Optional: Visual indicator that robot is speaking
                    if self.tts.is_speaking():
                        cv2.putText(frame, "SPEAKING...", (60, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
                        
                    cv2.imshow("Robot Vision", frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # 2. PREVENT SELF-LISTENING
                # If the robot is talking, we assume any sound is its own voice.
                # We clear the audio buffer so it doesn't hear itself.
                if self.tts.is_speaking():
                    self.whisper.data_queue.queue.clear()
                    continue

                # 3. HEAR
                text = self.whisper.process_audio()
                
                if text is not None:
                    current_transcription = self.whisper.get_transcription()
                    current_length = len(current_transcription)
                    
                    if current_length > last_transcription_length and current_length > 0:
                        user_input = current_transcription[-1].strip()
                        
                        if user_input and user_input not in processed_phrases:
                            print(f"🗣️  Heard: {user_input}")
                            
                            # 4. ACT
                            if is_looking:
                                print("✅ User is looking. Responding...")
                                print("🤖 Assistant: ", end="", flush=True)
                                
                                def stream_callback(text_chunk: str):
                                    print(text_chunk, end="", flush=True)
                                    self.tts.speak_streaming(text_chunk)
                                
                                # Because TTS is now threaded, this finishes quickly!
                                self.llm.generate(user_input, stream_callback=stream_callback)
                                self.tts.flush()
                                print("\n")
                            else:
                                print("❌ User is NOT looking. Ignoring.")
                            
                            processed_phrases.add(user_input)
                    
                    last_transcription_length = current_length
                    
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.whisper.stop_listening()
            self.gaze.release()
            print("Goodbye!")

if __name__ == "__main__":
    agent = VoiceAgent()
    agent.run()