from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

class CitationQualityMetric(BaseMetric):
    def __init__(self, 
                 threshold: float = 0.5,
                 include_reason: bool = True):
        self.threshold = threshold
        self.include_reason = include_reason
        self.score = 0
        self.success = False
        self.reason = None

    def measure(self, test_case: LLMTestCase):
        expected = set(str(e) for e in test_case.additional_metadata.get("expected_pages", []))
        retrieved = set(str(r) for r in test_case.additional_metadata.get("rag_pages", []))

        relevant_matches = len(expected & retrieved)

        # Recall
        if not expected:
            recall = 1.0 if not retrieved else 0.0
        else:
            recall = relevant_matches / len(expected)
        
        # Precision
        if not retrieved:
            precision = 0.0 
        else:
            precision = relevant_matches / len(retrieved)

        if (precision + recall) == 0:
            self.score = 0.0
        else:
            self.score = (2 * precision * recall) / (precision + recall)

        self.success = self.score >= self.threshold

        if self.include_reason:
            self.reason = (
                f"F1 Score: {self.score:.2f} (Precision: {precision:.2f}, Recall: {recall:.2f}). "
                f"Expected {expected}, but retrieved {retrieved}."
            )

        return self.score
    
    async def a_measure(self, test_case: LLMTestCase):
        return self.measure(test_case)
    
    def is_successful(self):
         return self.success

    @property
    def __name__(self):
        return "Citation Quality (F1)"