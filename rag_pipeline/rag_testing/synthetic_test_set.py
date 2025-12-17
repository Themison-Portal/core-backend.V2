import psycopg2
from dotenv import load_dotenv
from sqlalchemy import text
from rag_pipeline.database import AsyncSessionLocal # Assumed to be configured
import asyncio
import os
from pathlib import Path

# ragas dataset
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.transforms.relationship_builders.cosine import CosineSimilarityBuilder
from ragas.testset.transforms import apply_transforms, Parallel

from ragas.testset.transforms.extractors import (HeadlinesExtractor, 
                                    KeyphrasesExtractor,
                                    NERExtractor,
                                    EmbeddingExtractor
                                    )

from ragas.testset.synthesizers import (
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer
)

from ragas.testset.synthesizers.prompts import (
    ThemesPersonasInput,
    ThemesPersonasMatchingPrompt,
)

from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType

from ragas.testset.persona import Persona

from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )

from ragas.testset import TestsetGenerator


# LLM
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
import openai

# EMBEDDING
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings


LLM_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"

script_dir = Path(__file__).resolve().parent 
OUTPUT_DATASETS_PATH = script_dir / "data"
OUTPUT_DATASETS_PATH.mkdir(parents=True, exist_ok=True)


# Load environment variables from .env
load_dotenv()

async def fetch_documents(k: int = 100):

    if k <= 0:
        raise Exception("k should be positive")

    async with AsyncSessionLocal() as session:
        sql = text(
            """
            SELECT dd.id, dd.document_id, dd.content, dd.page_number
            FROM document_chunks_docling dd
            LIMIT :k
            """
            )
        
    
        result = await session.execute(
            sql,
            {"k": k}
        )

        rows = result.fetchall()

    docs = []
    for row in rows:
        docs.append({
            "document_chunk_content": row.content,
            "metadata": {
                "document_id": str(row.document_id),
                "page_number": row.page_number
            }
        })
    
    return docs


def get_pipeline_components():
    # LLM
    chat_model = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.1)
    generator_llm = LangchainLLMWrapper(chat_model)
    openai_client = openai.OpenAI()

    # EMBEDDING
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return generator_embeddings, generator_llm, openai_client

async def generate_test_set(input_size: int, output_size: int, dataset_output_name: str):
    docs = await fetch_documents(k=input_size)

    generator_embeddings, generator_llm, openai_client = get_pipeline_components()


    # KNOWLEDGE GRAPH
    kg = KnowledgeGraph()

    # we split the data into chunks
    # TODO (optional): make the splitting more relevant to our domain
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc["document_chunk_content"], "document_metadata": doc["metadata"]}
            )
        )

    # EXTRACTION: get information from each node, that is relevant to producing some relationship between them
    # TODO (optional): write custom extractors, for our domain
    # extractors take elements from each node (some specific elements)
    # and based on this establish relationships between graph nodes
    # we can create our custom relationships builders

    # headline_extractor = HeadlinesExtractor(llm=generator_llm, max_num=20)
    # headline_splitter = HeadlineSplitter(max_tokens=1500)
    
    # extracts keywords
    keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm, property_name="keyphrases")
    # extracts named entities
    ner_extractor = NERExtractor(llm=generator_llm, property_name="entities")
    # extract embedding of the given page_content (needed for CosineSimilarityBuilder)
    embedd_extractor = EmbeddingExtractor(embedding_model=generator_embeddings)

    # build the graph using extracted data (if only extractors, there is no edges)
    # by default relationship building is sequential,
    # if we want to establish edges on independent builders we use Parallel
    # to establish edges we need to introduce a notion of similarity,

    # # we will build edges between jaccard similar objects (jaccard in terms of NER)
    # # the name of this relationship is jaccard_similarity
    # rel_builder = JaccardSimilarityBuilder(property_name="entities", new_property_name="jaccard_similarity")

    rel_builder = CosineSimilarityBuilder(threshold=0.5)

    # TODO: we can add more relationships, to enable abstract queries
    
    transforms = [
        Parallel(
            keyphrase_extractor,
            ner_extractor,
            embedd_extractor
        ),
        rel_builder
    ]

    apply_transforms(kg, transforms=transforms)

    persona_young_nurse = Persona(
        name="Young Nurse",
        role_description="A young nurse that has just began working in the hostpital. needs clear and short reponses.",
    )

    persons_experienced_doctor = Persona(
        name="Experienced Doctor",
        role_description="An old experienced doctor, that needs detailed responses supported by medical knowledge."
    )

    personas = [persona_young_nurse, persons_experienced_doctor]


    # for multihop we need to have relationships
    if (len(kg.relationships) == 0):
        raise Exception("Multi-hop requires at least one relationship")
    

    query_distibution = [
        (
            SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"),
            0.5,
        ),
        (
            SingleHopSpecificQuerySynthesizer(
                llm=generator_llm, property_name="entities"
            ),
            0.5,
        ),
        # # to use multi-hop a relationship needs to exist
        # (
        #     MultiHopSpecificQuerySynthesizer(llm=generator_llm, property_name="cosine_similarity"),
        #     0.6,
        # )
        #  (
        #     MultiHopAbstractQuerySynthesizer(llm=generator_llm),
        #     0.3,
        # ),

    ]

    # testset generation

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )

    testset = generator.generate(testset_size=output_size, query_distribution=query_distibution)
    df = testset.to_pandas()

    df.to_csv(OUTPUT_DATASETS_PATH / dataset_output_name)

    print(f"File saved successfully as {OUTPUT_DATASETS_PATH / dataset_output_name}")


async def main():
    await generate_test_set(input_size=40, output_size=20, dataset_output_name="test.csv")


if __name__ == "__main__":
    asyncio.run(main())