import rclpy
from rclpy.time import Time
import pdb

class ROSData:
    def __init__(self, timeout: int = 3, queue_size: int = 1, name: str = ""):
        self.timeout = timeout
        self.last_time_received = None
        self.queue_size = queue_size
        self.data = None
        self.name = name
        self.phantom = False
    
    def get(self):
        return self.data
    
    def set(self, data):
        current_time = rclpy.clock.Clock().now()

        time_waited = float('inf')
        if self.last_time_received is not None:
            # Update time_waited with the actual time waited, otherwise stick to a huge value. This resets the data
            time_waited = (current_time - Time(seconds=self.last_time_received)).nanoseconds / 1e9  # Convert to seconds

        if self.queue_size == 1:
            self.data = data
        else:
            if self.data is None or time_waited > self.timeout:  # Reset queue if timeout
                self.data = []
            if len(self.data) == self.queue_size:
                self.data.pop(0)
            self.data.append(data)
        self.last_time_received = current_time.nanoseconds / 1e9  # Store time in seconds
        
    def is_valid(self, verbose: bool = False):
        current_time = rclpy.clock.Clock().now()

        time_waited = float('inf')
        if self.last_time_received is not None:
            # Update time_waited with the actual time waited, otherwise stick to a huge value. This resets the data
            time_waited = (current_time - Time(seconds=self.last_time_received)).nanoseconds / 1e9  # Convert to seconds

        valid = time_waited < self.timeout
        if self.queue_size > 1:
            valid = valid and len(self.data) == self.queue_size
        if verbose and not valid:
            print(f"Not receiving {self.name} data for {time_waited:.2f} seconds (timeout: {self.timeout} seconds)")
        return valid