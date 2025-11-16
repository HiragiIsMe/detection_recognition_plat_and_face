# utils/loading.py
import sys
import time
import threading

class LoadingAnimation:
    def __init__(self, message="Loading"):
        self.message = message
        self.loading = True
        self.animation_chars = ["/", "-", "\\", "|"]
        self.current_char = 0
        self.thread = None
        
    def _animate(self):
        while self.loading:
            sys.stdout.write(f'\r{self.message} {self.animation_chars[self.current_char]}')
            sys.stdout.flush()
            self.current_char = (self.current_char + 1) % len(self.animation_chars)
            time.sleep(0.1)
    
    def start(self):
        self.loading = True
        self.thread = threading.Thread(target=self._animate)
        self.thread.daemon = True
        self.thread.start()
    
    def stop(self, success_message=None):
        self.loading = False
        if self.thread:
            self.thread.join()
        sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear line
        if success_message:
            print(f"✅ {success_message}")

def simple_loading(message="Loading", duration=2):
    """Simple loading tanpa threading"""
    animation_chars = ["/", "-", "\\", "|"]
    start_time = time.time()
    
    while time.time() - start_time < duration:
        for char in animation_chars:
            sys.stdout.write(f'\r{message} {char}')
            sys.stdout.flush()
            time.sleep(0.1)
    
    sys.stdout.write('\r' + ' ' * 50 + '\r')  # Clear line