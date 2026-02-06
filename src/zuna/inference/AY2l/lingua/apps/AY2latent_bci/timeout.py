import signal
import platform
import os # Optional: for logging pid

# 1. Define a specific exception for timeouts
class TimeoutError(Exception):
    """Custom exception raised when a timeout occurs."""
    pass

# 2. Create the context manager class
class Timeout:
    """
    A context manager to raise a TimeoutError if the enclosed block
    takes longer than a specified duration.

    Uses signal.alarm() and is therefore *not compatible with Windows*.
    """
    def __init__(self, seconds=1, error_message='Timeout occurred'):
        if platform.system() == "Windows":
            raise RuntimeError("Timeout context manager is not compatible with Windows "
                               "due to reliance on signal.alarm().")
        if not isinstance(seconds, int) or seconds <= 0:
            raise ValueError("Timeout duration must be a positive integer number of seconds.")

        self.seconds = seconds
        self.error_message = error_message
        self._previous_handler = None

    def _handle_timeout(self, signum, frame):
        """Signal handler that raises the TimeoutError."""
        # Optional: Log the process ID where the timeout happened for debugging
        # print(f"!!! Timeout triggered in process PID: {os.getpid()} !!!")
        raise TimeoutError(self.error_message)

    def __enter__(self):
        # Register the signal handler for SIGALRM
        self._previous_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        # Set the alarm
        signal.alarm(self.seconds)
        # Return self allows 'with Timeout(..) as t:' usage if ever needed
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # --- Crucially: Cancel the alarm ---
        # This prevents the alarm from firing if the block finished on time.
        signal.alarm(0)

        # --- Restore the previous signal handler ---
        # This is important if timeouts are nested or other parts use SIGALRM.
        signal.signal(signal.SIGALRM, self._previous_handler)

        # --- Exception Handling ---
        # If the block exited due to our TimeoutError, the signal handler already
        # raised it. We don't need to do anything special here.
        # If the block exited due to *another* exception, we want that to propagate.
        # If the block exited normally (no exception), exc_type will be None.
        # Returning False (or None) from __exit__ ensures any exception
        # that occurred *inside* the 'with' block (including our TimeoutError)
        # is re-raised outside the block.
        return False # Do not suppress exceptions