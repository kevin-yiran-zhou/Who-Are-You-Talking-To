import threading
import time
import cv2
from robot_state import RobotState

# Import your existing modules
from gaze_detector import GazeDetector
from speech_to_text import Whisper
from language_model import LanguageModel
from text_to_speech import TextToSpeech

class AsyncVoiceAgent:
    def __init__(self):
        print("Initializing Independent Modules...")
        self.state = RobotState()
        
        # Initialize Hardware
        self.gaze = GazeDetector()
        self.whisper = Whisper()
        self.llm = LanguageModel(verbose=False)
        self.tts = TextToSpeech() # Already threaded from our previous fix!

    # --- WORKER 1: THE EARS (Background Thread) ---
    def ear_worker(self):
        print("👂 Ears listening...")
        self.whisper.start_listening()
        last_transcription_length = 0
        processed_phrases = set()

        while self.state.should_continue():
            # 1. Check if we should ignore audio (Self-Listening)
            if self.tts.is_speaking():
                self.whisper.data_queue.queue.clear() # Dump buffer
                time.sleep(0.1)
                continue

            # 2. Process Audio
            text = self.whisper.process_audio()
            
            if text:
                current = self.whisper.get_transcription()
                if len(current) > last_transcription_length:
                    phrase = current[-1].strip()
                    
                    if phrase and phrase not in processed_phrases:
                        message = f"You: {phrase}"
                        print(f"🗣️  {message}")
                        self.state.add_display_message(message)
                        # Push to queue for the Brain to handle later
                        self.state.heard_queue.put(phrase) 
                        processed_phrases.add(phrase)
                        
                    last_transcription_length = len(current)
            else:
                time.sleep(0.1) # Rest CPU
        
        self.whisper.stop_listening()

    # --- WORKER 2: THE BRAIN (Background Thread) ---
    def brain_worker(self):
        print("🧠 Brain active...")
        while self.state.should_continue():
            try:
                # 1. Wait for input (Blocking wait, efficient!)
                # Timeout allows us to check 'stop_event' periodically
                user_input = self.state.heard_queue.get(timeout=1.0)
                
                # 2. Check Context (Gaze)
                if self.state.is_looking:
                    print(f"✅ Processing: {user_input}")
                    
                    # Track assistant response
                    assistant_response = ""
                    
                    # Define how to send data to the Mouth
                    def stream_callback(text_chunk):
                        # Send directly to TTS queue
                        self.tts.speak_streaming(text_chunk) 
                        print(text_chunk, end="", flush=True)
                        nonlocal assistant_response
                        assistant_response += text_chunk

                    print("🤖 Assistant: ", end="", flush=True)
                    self.llm.generate(user_input, stream_callback=stream_callback)
                    self.tts.flush()
                    print("\n")
                    
                    # Add assistant response to display
                    if assistant_response.strip():
                        self.state.add_display_message(f"Assistant: {assistant_response.strip()}")
                else:
                    message = f"Ignored (Not Looking): {user_input}"
                    print(f"❌ {message}")
                    self.state.add_display_message(message)

                self.state.heard_queue.task_done()
                
            except threading.ThreadError:
                pass # Queue was empty
            except Exception:
                pass # Timeout or other error

    # --- MAIN CONTROLLER ---
    def run(self):
        # Start Background Threads
        ear_thread = threading.Thread(target=self.ear_worker, daemon=True)
        brain_thread = threading.Thread(target=self.brain_worker, daemon=True)
        
        ear_thread.start()
        brain_thread.start()

        print("\n=== SYSTEM RUNNING INDEPENDENTLY ===")
        print("Main thread: Handling Video")
        print("Thread 1:    Handling Audio")
        print("Thread 2:    Handling Logic")
        print("====================================\n")

        try:
            # Create window and set to fullscreen
            cv2.namedWindow("Robot Vision", cv2.WINDOW_NORMAL)
            cv2.setWindowProperty("Robot Vision", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            
            # --- MAIN THREAD: THE EYES (UI Loop) ---
            # We keep Video in the main thread because OpenCV GUIs 
            # often crash if run in background threads.
            while True:
                frame, is_looking = self.gaze.process_frame()
                
                # Update Shared State immediately
                self.state.is_looking = is_looking
                
                # Update UI
                if frame is not None:
                    color = (0, 255, 0) if is_looking else (0, 0, 255)
                    cv2.circle(frame, (30, 30), 15, color, -1)
                    
                    if self.tts.is_speaking():
                        cv2.putText(frame, "SPEAKING...", (60, 35), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
                    
                    # Display conversation messages
                    messages = self.state.get_display_messages()
                    if messages:
                        # Display messages in bottom-left corner
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.5
                        thickness = 1
                        line_height = 22
                        start_x = 10
                        max_width = frame.shape[1] - 30
                        
                        # Calculate total height needed (with wrapping)
                        total_lines = 0
                        wrapped_messages = []
                        for msg in messages:
                            # Simple wrapping: split by character count
                            max_chars_per_line = int(max_width / (font_scale * 10))
                            words = msg.split()
                            lines = []
                            current_line = ""
                            for word in words:
                                test_line = current_line + (" " if current_line else "") + word
                                if len(test_line) > max_chars_per_line and current_line:
                                    lines.append(current_line)
                                    current_line = word
                                else:
                                    current_line = test_line
                            if current_line:
                                lines.append(current_line)
                            wrapped_messages.append(lines)
                            total_lines += len(lines)
                        
                        start_y = frame.shape[0] - (total_lines * line_height) - 10
                        
                        # Draw messages
                        current_y = start_y
                        for i, (msg, lines) in enumerate(zip(messages, wrapped_messages)):
                            # Color code: green for "You:", yellow for "Assistant:", red for "Ignored"
                            if msg.startswith("You:"):
                                text_color = (0, 255, 0)  # Green
                            elif msg.startswith("Assistant:"):
                                text_color = (255, 200, 0)  # Yellow/Orange
                            else:
                                text_color = (0, 0, 255)  # Red
                            
                            # Draw each line of the wrapped message
                            for line in lines:
                                cv2.putText(frame, line, (start_x, current_y), 
                                          font, font_scale, text_color, thickness)
                                current_y += line_height

                    cv2.imshow("Robot Vision", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.state.stop() # Signal threads to die
            ear_thread.join()
            brain_thread.join()
            self.gaze.release()
            self.tts.stop()
            print("System Shutdown.")

if __name__ == "__main__":
    agent = AsyncVoiceAgent()
    agent.run()