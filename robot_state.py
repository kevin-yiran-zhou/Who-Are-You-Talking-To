import queue
import threading

class RobotState:
    def __init__(self):
        # 1. SHARED DATA (The "Bulletin Board")
        self.is_looking = False  # Updated by Gaze
        self.is_speaking = False # Updated by TTS
        
        # 2. COMMUNICATION QUEUES
        # Ears -> Brain
        self.heard_queue = queue.Queue() 
        
        # Brain -> Mouth
        self.speech_queue = queue.Queue()
        
        # 3. CONTROL SIGNALS
        self.stop_event = threading.Event()
        
        # 4. DISPLAY MESSAGES (Thread-safe)
        self.display_messages = []  # List of message strings to display
        self.display_lock = threading.Lock()  # Lock for thread-safe access
        self.max_display_messages = 2  # Maximum number of messages to keep

    def stop(self):
        """Signal all threads to shut down."""
        self.stop_event.set()
        
    def should_continue(self):
        return not self.stop_event.is_set()
    
    def add_display_message(self, message: str):
        """Add a message to the display list (thread-safe)."""
        with self.display_lock:
            self.display_messages.append(message)
            # Keep only the most recent messages
            if len(self.display_messages) > self.max_display_messages:
                self.display_messages.pop(0)
    
    def get_display_messages(self):
        """Get a copy of the display messages (thread-safe)."""
        with self.display_lock:
            return self.display_messages.copy()