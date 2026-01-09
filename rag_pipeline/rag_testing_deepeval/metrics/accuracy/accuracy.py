from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from rag_pipeline.rag_testing_deepeval.metrics.accuracy.scorer import TextAccuracyScorer


LLM_MODEL_NAME = "gpt-4o-mini"

class AnswerAccuracyMetric(BaseMetric):

    def __init__(self, 
                 evaluation_model: str,
        threshold: float = 1.0,
        include_reason: bool = True,
        strict_mode: bool = True,
        async_mode: bool = True):
        
        self.threshold = threshold
        # Optional
        self.evaluation_model = evaluation_model
        self.include_reason = include_reason
        self.strict_mode = strict_mode
        self.async_mode = async_mode


    

    def measure(self, test_case: LLMTestCase) -> float:
        try:
            scorer = TextAccuracyScorer(evaluation_model=self.evaluation_model)
            result = scorer(question=test_case.input,
                            reference_answer=test_case.expected_output,
                            text_to_score=test_case.actual_output)
            
            self.score = result.score
            if self.include_reason:
                self.reason =result.reasoning
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
        return "Accuracy Metric (dspy llm as judge)"