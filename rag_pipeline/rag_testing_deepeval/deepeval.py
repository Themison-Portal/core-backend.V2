# deepeval
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
# python stuff
from pydantic import BaseModel, ValidationError
from typing import List
from enum import Enum
import time
import asyncio
# testing utils
from rag_pipeline.rag_testing_deepeval.testcases import test_data
# rag functions
from rag_pipeline.query_data_store import rag_query

# custom metrics
from rag_pipeline.rag_testing_deepeval.metrics.accuracy.accuracy import AnswerAccuracyMetric
from rag_pipeline.rag_testing_deepeval.metrics.citation.citation import CitationQualityMetric
from rag_pipeline.rag_testing_deepeval.metrics.latency.latency import LatencyMetric
from rag_pipeline.rag_testing_deepeval.metrics.completeness.completeness import CompletenessMetric


class RagSource(BaseModel):
    section: str
    page: int
    filename: str
    exactText: str
    chunk_index: int
    relevance: str
    context: str
    highlightURL: str

class RagStructuredResponse(BaseModel):
    response: str
    sources: List[RagSource]

class Metrics(Enum):
    """parent for other metrics enums"""
    pass

class LLMMetrics(Metrics):
    RELEVANCY = "relevancy"
    FAITHFULNESS = "faithfulness"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"

class StandardMetrics(Metrics):
    CITATION = "citation"
    LATENCY = "latency"

# class LatencyResult(BaseModel):
#     """Stores latency measurements for RAG pipeline"""
#     query: str
#     total_time_ms: float
#     avg_time_ms: float = None

LLM_MODEL_NAME = "gpt-4o-mini"

class TestPipeline:
    def __init__(self, rag_function):
        self.rag_function = rag_function
        self.latency_results = []
        # self.citations = []
    
    def __is_valid_response(self, data: dict) -> bool:
        return (
            isinstance(data, dict) and 
            "response" in data and 
            "sources" in data and
            isinstance(data["sources"], list)
        )
    
    async def __run_rag(self, query: str, measure_latency: bool = False):
        """
        Runs async rag pipeline and returns structure that deepeval can utilize for testing.
        Optionally measures latency.
        
        Returns:
            tuple: (response, contexts, sources, latency_ms or None)
        """
        start_time = time.perf_counter() if measure_latency else None
        
        result = await self.rag_function(query)
        
        end_time = time.perf_counter() if measure_latency else None
        latency_ms = (end_time - start_time) * 1000 if measure_latency else None
        
        if not self.__is_valid_response(result):
            raise Exception("RAG response should be of RagStructuredResponse structure")
        
        contexts = [source.exactText for source in result["sources"]]
        return (result["response"], contexts, result["sources"], latency_ms)
    
    async def __create_test_cases(self, test_data: list, measure_latency: bool = False):
        """Creates test cases asynchronously and optionally measures latency"""
        test_cases = []
        
        for data in test_data:
            result, contexts, sources, latency = await self.__run_rag(
                query=data["input"], 
                measure_latency=measure_latency
            )
            
            # if measure_latency and latency is not None:
            #     self.latency_results.append(
            #         LatencyResult(query=data["input"], total_time_ms=latency)
            #     )
            
            test_case = LLMTestCase(
                input=data["input"],
                actual_output=result,
                retrieval_context=contexts,
                expected_output=data["expected_output"],
                additional_metadata= {
                    "expected_pages": data["pages"],
                    "rag_pages": [s.page for s in sources],
                    "latency": latency
                }
            )
            test_cases.append(test_case)
        
        return test_cases
    
   
    async def __create_test_cases_concurrent(
        self, 
        test_data: list,
        measure_latency: bool = False, 
        max_concurrent: int = 5
    ):
        """
        Creates test cases with concurrent execution for faster testing.
        
        Args:
            test_data: List of test case dictionaries
            max_concurrent: Maximum number of concurrent requests
        """
        test_cases = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_test_case(data):
            async with semaphore:
                result, contexts, sources, latency = await self.__run_rag(
                    query=data["input"], 
                    measure_latency=measure_latency
                )
                
                # if measure_latency and latency is not None:
                #     self.latency_results.append(
                #         LatencyResult(query=data["input"], total_time_ms=latency)
                #     )
                
                return LLMTestCase(
                    input=data["input"],
                    actual_output=result,
                    retrieval_context=contexts,
                    expected_output=data["expected_output"],                    
                    additional_metadata= {
                        "expected_pages": data["pages"],
                        "rag_pages": [s.page for s in sources],
                        "latency": latency
                    }
                )
        
        test_cases = await asyncio.gather(*[process_test_case(data) for data in test_data])
        return test_cases
    
    def __get_metric_instance(self, metric: Metrics):
        """Maps enum to actual metric instances"""
        metric_map = {
            # tests relevancy of output based on the contexts
            LLMMetrics.RELEVANCY: AnswerRelevancyMetric(threshold=0.8, model=LLM_MODEL_NAME),
            # penalizes hallucination and any contradicting information from the context
            LLMMetrics.FAITHFULNESS: FaithfulnessMetric(strict_mode=True, model=LLM_MODEL_NAME),
            # our custom metric for accuracy evaluation
            LLMMetrics.ACCURACY: AnswerAccuracyMetric(evaluation_model=LLM_MODEL_NAME),
            # custom metric for completeness
            LLMMetrics.COMPLETENESS: CompletenessMetric(evaluation_model=LLM_MODEL_NAME),
            # citation
            StandardMetrics.CITATION: CitationQualityMetric(),
            # latency
            StandardMetrics.LATENCY: LatencyMetric()
        }
        return metric_map.get(metric)
    
    async def evaluate_rag(
        self, 
        test_data: list, 
        criteria: List[Metrics],
        concurrent: bool = False,
        max_concurrent: int = 5
    ):
        """
        Evaluates RAG pipeline with specified metrics.
        
        Args:
            test_data: List containing test cases
            criteria: List of LLMMetrics to evaluate
            measure_latency: Whether to measure and report latency
            concurrent: Whether to run test cases concurrently (faster but may affect latency measurements)
            max_concurrent: Maximum concurrent requests when concurrent=True
        """
        if not all(isinstance(c, Metrics) for c in criteria):
            raise Exception("All criteria must be LLMMetrics enum values")
        
        # TODO: probably a type for test data could be useful
        if not isinstance(test_data, list):
            raise Exception("test_data must be a list of test cases")
        
        self.latency_results = []

        measure_latency = StandardMetrics.LATENCY in criteria
        
        if concurrent:
            tests = await self.__create_test_cases_concurrent(
                test_data, 
                measure_latency=measure_latency,
                max_concurrent=max_concurrent
            )
        else:
            tests = await self.__create_test_cases(test_data, measure_latency)
        
        # Convert enum to actual metric instances
        metrics = [self.__get_metric_instance(m) for m in criteria]
        
        # Run deepeval evaluation
        print("\n=== Running LLM Metrics Evaluation ===")
        evaluate(tests, metrics=metrics)
        
        # # Report latency if measured
        # if measure_latency and self.latency_results:
        #     self.__report_latency()
    
    # def __report_latency(self):
    #     """Prints latency statistics"""
    #     print("\n=== Latency Results ===")
        
    #     total_times = [r.total_time_ms for r in self.latency_results]
    #     avg_latency = sum(total_times) / len(total_times)
    #     min_latency = min(total_times)
    #     max_latency = max(total_times)
        
    #     print(f"Average Latency: {avg_latency:.2f}ms")
    #     print(f"Min Latency: {min_latency:.2f}ms")
    #     print(f"Max Latency: {max_latency:.2f}ms")
    #     print(f"\nPer-Query Latency:")
        
    #     for result in self.latency_results:
    #         query_preview = result.query[:50] + "..." if len(result.query) > 50 else result.query
    #         print(f"  {result.total_time_ms:.2f}ms - {query_preview}")

    
    # async def benchmark_latency(
    #     self, 
    #     query: str, 
    #     num_runs: int = 10,
    #     warmup_runs: int = 2
    # ):
    #     """
    #     Runs latency benchmark for a single query multiple times.
        
    #     Args:
    #         query: Query string to benchmark
    #         num_runs: Number of times to run the query
    #         warmup_runs: Number of warmup runs (not included in statistics)
    #     """
    #     print(f"\n=== Benchmarking Query (warmup={warmup_runs}, n={num_runs}) ===")
    #     print(f"Query: {query}\n")
        
    #     # Warmup runs
    #     if warmup_runs > 0:
    #         print(f"Running {warmup_runs} warmup runs...")
    #         for i in range(warmup_runs):
    #             await self.__run_rag(query, measure_latency=False)
    #         print("Warmup complete.\n")
        
    #     # Actual benchmark runs
    #     latencies = []
        
    #     for i in range(num_runs):
    #         _, _, _, latency = await self.__run_rag(query, measure_latency=True)
    #         latencies.append(latency)
    #         print(f"Run {i+1}: {latency:.2f}ms")
        
    #     avg = sum(latencies) / len(latencies)
    #     std_dev = (sum((x - avg) ** 2 for x in latencies) / len(latencies)) ** 0.5
        
    #     print(f"\nAverage: {avg:.2f}ms")
    #     print(f"Min: {min(latencies):.2f}ms")
    #     print(f"Max: {max(latencies):.2f}ms")
    #     print(f"Std Dev: {std_dev:.2f}ms")
    #     print(f"P50: {sorted(latencies)[len(latencies)//2]:.2f}ms")
    #     print(f"P95: {sorted(latencies)[int(len(latencies)*0.95)]:.2f}ms")


async def main():
    testing_pipeline = TestPipeline(rag_function=rag_query)
    
    await testing_pipeline.evaluate_rag(
        test_data=test_data,
        criteria=[LLMMetrics.ACCURACY, 
                  LLMMetrics.COMPLETENESS, 
                  LLMMetrics.FAITHFULNESS, 
                  LLMMetrics.RELEVANCY,
                  StandardMetrics.CITATION,
                  StandardMetrics.LATENCY],
    )
    

if __name__ == "__main__":
    asyncio.run(main())