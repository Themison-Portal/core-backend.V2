from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from rag_pipeline.rag_testing_deepeval.metrics.completeness.scorer import TestCompletenessScorer



class CompletenessMetric(BaseMetric):

    def __init__(self, 
                 evaluation_model: str,
        threshold: float = 0.1,
        include_reason: bool = True,
        strict_mode: bool = True,
        async_mode: bool = True):
        
        self.threshold = threshold
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode

        self.evaluation_model = evaluation_model

    

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            scorer = TestCompletenessScorer(evaluation_model=self.evaluation_model)
            result = scorer(reference_text=test_case.expected_output,
                            target_text=test_case.actual_output
                            )
            
            # all parts
            all_parts = result.num_parts
            # rag covered parts
            rag_covered_parts = result.num_covered
            
            
            self.score = 1.0 if all_parts == 0 else rag_covered_parts / all_parts
            if self.include_reason:
                self.reason = result.coverage_analysis
            self.success = self.score >= self.threshold
            return self.score
        except Exception as e:
            # set metric error and re-raise it
            self.error = str(e)
            raise

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)
    
    def is_successful(self) -> bool:
        if self.error is not None:
            self.success = False
        else:
            return self.success
        
    @property
    def __name__(self):
        return "Completeness Metric (dspy llm as judge)"