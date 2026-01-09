import asyncio
from typing import List
from rag_pipeline.query_data_store_biobert import rag_query_biobert
from rag_pipeline.query_data_store import rag_query
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import pandas as pd
from ragas import evaluate
import asyncio
from datasets import Dataset 
from enum import StrEnum
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio 

from ragas.metrics import (
    context_precision, 
    faithfulness, 
    context_recall,
    answer_relevancy
)

# LLM
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import openai

# # EMBEDDING
# from ragas.embeddings import LangchainEmbeddingsWrapper
# from langchain_community.embeddings import HuggingFaceEmbeddings


LLM_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"


class ModelName(StrEnum):
    GPT_4o = "gpt-4o-mini"

class EmbeddName(StrEnum):
    BIOBERT = "pritamdeka/S-BioBert-snli-multinli-stsb"
    OPENAI = "text-embedding-3-small"


load_dotenv()

script_dir = Path(__file__).resolve().parent 
OUTPUT_TESTING = script_dir / "test_results"
OUTPUT_TESTING.mkdir(parents=True, exist_ok=True)

TEST_SET = script_dir / "synthetic_dataset" / "data" / "test2.csv"



class TestPipeline:

    
    def __init__(self):
        """
        to_test: List of lists of params for a grid search to make on.
        each array is a list of parameter values
        """

        self.test_set = self.__load_test_set()

        chat_model = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.1)
        self.llm_judge = LangchainLLMWrapper(chat_model)

        # # EMBEDDING
        # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        # self.judge_embedding = LangchainEmbeddingsWrapper(embeddings)

    def __load_test_set(self):
        if not TEST_SET.exists():
            raise FileNotFoundError(f"Test set not found {TEST_SET}")
        return pd.read_csv(TEST_SET)
    

    async def __process_single_query(self, query: str, model: str, embedding: str, semaphore: asyncio.Semaphore):
        """
        Helper function to process a single query with rate limiting (semaphore).
        """
        async with semaphore:  # Limits how many of these run at once
            try:
                if model == ModelName.GPT_4o:
                    if embedding == EmbeddName.BIOBERT:
                        result = await rag_query_biobert(query)
                    elif embedding == EmbeddName.OPENAI:
                        result = await rag_query(query)
                    else:
                        raise Exception("Incorrect embedding type")
                else:
                    raise Exception("Incorrect model")
                
                answer = result.get("response", "")
                raw_sources = result.get("sources", [])
                context_strings = [s.context for s in raw_sources if hasattr(s, 'context')]
                
                return answer, context_strings

            except Exception as e:
                print(f"Error processing query '{query}': {e}")
                return "Error producing answer", ["No context due to error"]

    async def __process_batch(self, model, embedding):
        """
        Executes rag queries concurrently.
        """
        queries = self.test_set["user_input"].tolist()
        
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)
        
        # Create a list of tasks
        tasks = [
            self.__process_single_query(query, model, embedding, semaphore)
            for query in queries
        ]
        
        # Run tasks concurrently with a progress bar
        # gather returns results in the same order as inputs
        results = await tqdm_asyncio.gather(*tasks, desc="Processing Queries")

        # Unzip results into separate lists
        answers, contexts = zip(*results)
        
        return list(answers), list(contexts)

    
    async def evaluate_rag(self):
        """
            Runs RAG system evaluation, gathers results.
        """

        serialized_average_results = []

        if 'user_input' not in self.test_set.columns or 'reference' not in self.test_set.columns:
            raise ValueError("CSV must contain 'user_input' and 'reference' columns")
        

        for model in ModelName:
            for embedd in EmbeddName:

                # run evaluation with specific model and embedding
                generated_answers, retrieved_contexts = await self.__process_batch(model, embedd)

                # eval dict
                data_dict = {
                    "user_input": self.test_set["user_input"].tolist(),
                    "retrieved_contexts": retrieved_contexts,
                    "response": generated_answers,
                    "reference": self.test_set["reference"].tolist()
                }

                dataset = Dataset.from_dict(data_dict)

                metrics = [
                    answer_relevancy, 
                    faithfulness
                ]
        
                results = evaluate(
                    llm=self.llm_judge,
                    embeddings=None, # no score uses embedding
                    dataset=dataset, 
                    metrics=metrics,
                )

                
        
                results_df = results.to_pandas()
                safe_name = str(f"{model}_{embedd}").replace("/", "__") 

                serialized_average_results.append((safe_name, results))

                results_df.to_csv(OUTPUT_TESTING / Path(f"{safe_name}_eval_res.csv"))

        
        return serialized_average_results
       


    

# # Averaged results
# [('pritamdeka__S-BioBert-snli-multinli-stsb', 
#   {'context_precision': 0.2500,
#     'faithfulness': 0.3718,
#     'context_recall': 0.0833}), 
# ('text-embedding-3-small', 
#  {'context_precision': 0.7375, 
#   'faithfulness': 0.6427, 
#   'context_recall': 0.5833})]
if __name__ == "__main__":
    testing = TestPipeline()
    results = asyncio.run(testing.evaluate_rag())
    # results = testing.evaluate_rag()
    print(results)
    # results = [('pritamdeka__S-BioBert-snli-multinli-stsb', {'context_precision': 0.2500, 'faithfulness': 0.3718, 'context_recall': 0.0833}), ('text-embedding-3-small', {'context_precision': 0.7375, 'faithfulness': 0.6427, 'context_recall': 0.5833})]
    with open(OUTPUT_TESTING / "averaged_results.txt", "w") as file:
        file.write(str(results))
        # for model, dict_val in results:
            # file.write("Configuration: " + model + "\n")
            # for k,v in dict_val.items():
            #     file.write(f"{k}: {v}" + "\n")
            # file.write("\n")

