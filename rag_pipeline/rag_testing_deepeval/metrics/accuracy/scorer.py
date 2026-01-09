import dspy



class AccuracyScorer(dspy.Signature):
    """Score the correctness of a text answer compared to a reference answer.
    
    Return:
    - 0 if the text is wrong/incorrect
    - 1 if the text is partially correct
    - 2 if the text is fully correct
    """
    
    question: str = dspy.InputField(desc="The question being answered")
    reference_answer: str = dspy.InputField(desc="The correct/reference answer")
    text_to_score: str = dspy.InputField(desc="The text answer to evaluate")
    score: int = dspy.OutputField(desc="Score: 0 (wrong), 1 (partially correct), or 2 (fully correct)")
    reasoning: str = dspy.OutputField(desc="Explanation for the score")


class TextAccuracyScorer(dspy.Module):
    def __init__(self, evaluation_model: str):
        super().__init__()
        self.evaluation_model = evaluation_model
        
        self.scorer = dspy.ChainOfThought(AccuracyScorer)
    
    def forward(self, question, reference_answer, text_to_score):

        with dspy.context(lm=dspy.LM(self.evaluation_model)):
            result = self.scorer(
                question=question,
                reference_answer=reference_answer,
                text_to_score=text_to_score
            )
        
        # Ensure score is an integer between 0-2
        try:
            score = int(result.score)
            if score not in [0, 1, 2]:
                score = 1  # Default to partially correct if invalid
        except (ValueError, AttributeError):
            score = 1
        
        return dspy.Prediction(
            score=score,
            reasoning=result.reasoning
        )
