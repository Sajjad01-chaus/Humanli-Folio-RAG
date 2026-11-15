"""Conservative token reservation for a single process/provider quota."""

import time
from collections.abc import Callable


class TokenBucket:
    def __init__(
        self,
        capacity: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if capacity <= 0:
            raise ValueError("Token capacity must be positive.")
        self.capacity = capacity
        self.available = float(capacity)
        self.clock, self.sleep = clock, sleep
        self.updated = clock()

    def reserve(self, tokens: int) -> None:
        if tokens > self.capacity:
            raise ValueError("Request exceeds the configured provider token budget.")
        while True:
            now = self.clock()
            self.available = min(self.capacity, self.available + (now - self.updated) * self.capacity / 60)
            self.updated = now
            if self.available >= tokens:
                self.available -= tokens
                return
            self.sleep((tokens - self.available) * 60 / self.capacity + 0.05)

    def refund(self, tokens: int) -> None:
        self.available = min(self.capacity, self.available + max(tokens, 0))
