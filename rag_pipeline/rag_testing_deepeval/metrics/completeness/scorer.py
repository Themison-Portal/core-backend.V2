import dspy
from typing import List

lm = dspy.LM('openai/gpt-4o-mini')
dspy.configure(lm=lm)

class PartIdentifier(dspy.Signature):
    """Identify distinct logical parts or sections in the given text.
    Each part should represent a separate topic, requirement, or piece of information.
    Return a numbered list of parts with brief descriptions."""
    
    text: str = dspy.InputField(desc="The source text to analyze for distinct parts")
    num_parts: int = dspy.OutputField(desc="Total number of distinct parts identified (as integer)")
    parts_list: List[str] = dspy.OutputField(desc="Numbered list of identified parts with descriptions")


class PartCoverageChecker(dspy.Signature):
    """Check how many parts from the reference text are covered in the target text.
    A part is covered if the target text contains the same information, even if phrased differently.
    A part is NOT covered if the target text hallucinates or invents information not in the reference.
    Return 0 if any hallucinations are detected."""
    
    reference_parts: List[str] = dspy.InputField(desc="List of parts from the reference text")
    target_text: str = dspy.InputField(desc="The text to check for coverage of reference parts")
    num_covered: int = dspy.OutputField(desc="Number of parts from reference that are accurately covered (0 if hallucinations detected)")
    coverage_analysis: str = dspy.OutputField(desc="Detailed explanation of which parts are covered and any hallucinations found")


class TestCompletenessScorer(dspy.Module):
    
    def __init__(self, evaluation_model: str):
        super().__init__()
        self.evaluation_model = evaluation_model
        # create thought processes
        self.identify_parts = dspy.ChainOfThought(PartIdentifier)
        self.check_coverage = dspy.ChainOfThought(PartCoverageChecker)
    
    def forward(self, reference_text, target_text):

        with dspy.context(lm=dspy.LM(self.evaluation_model)):
            # Step 1: Identify parts in reference_text
            part_result = self.identify_parts(text=reference_text)
            parts_list = part_result.parts_list
            
            # Step 2: Check coverage in target_text
            coverage_result = self.check_coverage(
                reference_parts=parts_list,
                target_text=target_text
            )
        
        num_parts = int(part_result.num_parts)
        num_covered = int(coverage_result.num_covered)
        
        return dspy.Prediction(
            num_parts=num_parts,
            parts_list=parts_list,
            num_covered=num_covered,
            coverage_analysis=coverage_result.coverage_analysis
        )
