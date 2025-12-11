from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
import time
import threading
from typing import Optional, Callable


class LanguageModel:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507-FP8", verbose: bool = True):
        """Initialize the language model.
        
        Args:
            model_name: Name of the model to load
            verbose: Whether to print loading messages
        """
        if verbose:
            print("Loading model...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        self.messages = []
        self.model_name = model_name
        
        # Add system message to keep responses concise (1-3 sentences)
        self.messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Keep your responses brief and concise - limit your answers to 1-3 sentences. This is for a voice interface, so be conversational but brief. NO EMOJIS ALLOWED."
        })
        
        if verbose:
            print("Model loaded!")
    
    def generate(self, user_input: str, max_new_tokens: int = 200, 
                 stream_callback: Optional[Callable[[str], None]] = None) -> str:
        """Generate a response to user input.
        
        Args:
            user_input: The user's input text
            max_new_tokens: Maximum number of tokens to generate (default: 150, enough for 1-3 sentences)
            stream_callback: Optional callback function called with each token as it's generated
            
        Returns:
            The generated response text
        """
        if not user_input.strip():
            return ""
        
        # Add user message to conversation history
        self.messages.append({"role": "user", "content": user_input})
        
        # Prepare the model input
        text = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        # Set up streaming
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        # Generate response with streaming
        generation_kwargs = dict(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer
        )
        
        # Start generation in a separate thread
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        # Stream the response token by token
        content = ""
        for new_text in streamer:
            if stream_callback:
                stream_callback(new_text)
            content += new_text
        
        thread.join()
        
        # Add assistant response to conversation history
        self.messages.append({"role": "assistant", "content": content})
        
        return content
    
    def reset(self):
        """Reset the conversation history (but keep the system message)."""
        # Reset to just the system message
        self.messages = [{
            "role": "system",
            "content": "You are a helpful assistant. Keep your responses brief and concise - limit your answers to 1-3 sentences. This is for a voice interface, so be conversational but brief. NO EMOJIS ALLOWED."
        }]
    
    def get_history(self):
        """Get the current conversation history."""
        return self.messages.copy()
    
    def run_interactive(self):
        """Run an interactive command-line chat session."""
        print("Type 'quit' or 'exit' to end the conversation.\n")
        
        while True:
            user_input = input("🗣️  You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            start_time = time.time()
            print(f"\n🤖 Assistant: ", end="", flush=True)
            
            response = self.generate(
                user_input,
                stream_callback=lambda text: print(text, end="", flush=True)
            )
            
            end_time = time.time()
            generation_time = end_time - start_time
            print(f"\n[Generation time: {generation_time:.2f} seconds]\n")


if __name__ == "__main__":
    # Run interactive mode if script is executed directly
    model = LanguageModel()
    model.run_interactive()