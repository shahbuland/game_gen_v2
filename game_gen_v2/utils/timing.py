import time

class Stopwatch:
    def __init__(self):
        self.start_time = None

    def reset(self):
        """Prime the stopwatch for measurement."""
        self.start_time = time.time()

    def hit(self, samples: int) -> float:
        """
        Measure the average time per 1000 samples since the last reset.

        Args:
            samples (int): The number of samples processed.

        Returns:
            float: The time in seconds per 1000 samples.
        """
        if self.start_time is None:
            raise ValueError("Stopwatch must be reset before calling hit.")

        elapsed_time = time.time() - self.start_time
        avg_time_per_sample = elapsed_time / samples
        return avg_time_per_sample * 1000  # Return time per 1000 samples