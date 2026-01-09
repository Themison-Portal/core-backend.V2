from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class LatencyMetric(BaseMetric):
    def __init__(self, 
                 threshold: float = 0.5,
                 threshold_ms: float = 10000,
                 include_reason: bool = True):
        self.threshold = threshold
        self.include_reason = include_reason
        self.threshold_ms = threshold_ms
        self.score = 0
        self.success = False
        self.reason = None

    def measure(self, test_case: LLMTestCase) -> float:
        """
        Measures latency from test case.
        Returns a score between 0 and 1 (1 = best performance).
        """

        latency = test_case.additional_metadata.get("latency", None)
        if latency is None:
            raise ValueError("Test case must have latency measured")
        
        
        # Calculate score based on threshold
        if latency <= self.threshold_ms:
            score = 1.0
        else:
            # Gradual degradation above threshold
            score = self.threshold_ms / latency
        
        self.score = score
        self.success = score >= self.threshold
        
        # Store the reason with actual latency value
        self.reason = f"Latency: {latency:.2f}ms (Threshold: {self.threshold_ms}ms)"
        
        return self.score
    
    async def a_measure(self, test_case: LLMTestCase) -> float:
        """Async version of measure"""
        return self.measure(test_case)
    
    def is_successful(self) -> bool:
        """Returns whether the test passed based on latency threshold"""
        return self.success
    
    @property
    def __name__(self):
        return "Latency"
