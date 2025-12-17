import asyncio
from rag_pipeline.query_data_store_biobert import rag_query_biobert
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import pandas as pd
from ragas import evaluate
from datasets import Dataset 


from ragas.metrics import (
    context_precision, 
    faithfulness, 
    context_recall,
)

# LLM
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import openai

# EMBEDDING
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings


LLM_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"



load_dotenv()


script_dir = Path(__file__).resolve().parent 
OUTPUT_DATASETS_PATH = script_dir / "data"

async def evaluate_rag(test_set):
    answers = []
    contexts = []

    for query in test_set["user_input"]:
        try:
            result = await rag_query_biobert(query)

            answer = result["response"]
            raw_sources = result["sources"]
            
            context_strings = [source.context for source in raw_sources]


            answers.append(answer)
            contexts.append(context_strings)

        except Exception as e:
            print(f"Error processing query '{query}': {e}")
            answers.append("Error")
            contexts.append([])

    return answers, contexts

async def main():

    df = pd.read_csv(OUTPUT_DATASETS_PATH / "test2.csv")


    chat_model = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.1)
    llm_judge = LangchainLLMWrapper(chat_model)
    openai_client = openai.OpenAI()

    # EMBEDDING
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    judge_embedding = LangchainEmbeddingsWrapper(embeddings)

    if 'user_input' not in df.columns or 'reference' not in df.columns:
        raise ValueError("CSV must contain 'user_input' and 'reference' columns")
    
    generated_answers, retrieved_contexts = await evaluate_rag(df)


    # eval dict
    data_dict = {
        "question": df["user_input"].tolist(),
        "answer": generated_answers,
        "contexts": retrieved_contexts,
        "ground_truth": df["reference"].tolist()
    }

    dataset = Dataset.from_dict(data_dict)

    metrics = [
        context_precision, 
        faithfulness,
        context_recall
    ]
    

    print("Starting RAGAS evaluation...")
    results = evaluate(
        llm=llm_judge,
        embeddings=judge_embedding,
        dataset=dataset, 
        metrics=metrics,
    )

    print("\nAverage Scores:")
    print(results)
    
    results_df = results.to_pandas()
    results_df.to_csv(OUTPUT_DATASETS_PATH / "ragas_evaluation_results.csv")
    print("\nDetailed results saved to 'ragas_evaluation_results.csv'")


#Average Scores:
#{'context_precision': 0.5000, 'faithfulness': 0.2455, 'context_recall': 0.1250}
if __name__ == "__main__":
    asyncio.run(main())