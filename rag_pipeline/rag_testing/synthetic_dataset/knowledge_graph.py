import psycopg2
from dotenv import load_dotenv
from sqlalchemy import text
from rag_pipeline.database import AsyncSessionLocal # Assumed to be configured
import asyncio
import os
from pathlib import Path
import pickle

# ragas dataset
from ragas.testset.transforms.relationship_builders.traditional import(
                                                                    OverlapScoreBuilder,
                                                                    JaccardSimilarityBuilder
                                                                )
from ragas.testset.transforms.relationship_builders.cosine import CosineSimilarityBuilder, SummaryCosineSimilarityBuilder
from ragas.testset.transforms import apply_transforms, Parallel

from ragas.testset.transforms.extractors import (HeadlinesExtractor, 
                                    KeyphrasesExtractor,
                                    NERExtractor,
                                    EmbeddingExtractor,
                                    SummaryExtractor,
                                    EmbeddingExtractor
                                    )

from ragas.testset.transforms.splitters import (HeadlineSplitter)


from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType


script_dir = Path(__file__).resolve().parent 
OUTPUT_DATASETS_PATH = script_dir / "data"
OUTPUT_DATASETS_PATH.mkdir(parents=True, exist_ok=True)

def apply_and_cache_stage(kg, transforms, stage_name, force_rebuild=False):
    cache_file = OUTPUT_DATASETS_PATH / f"kg_{stage_name}.pkl"
    
    if not force_rebuild and cache_file.exists():
        with open(cache_file, "rb") as f:
            kg = pickle.load(f)
        print(f"✅ Loaded {stage_name} cache")
        return kg
    
    apply_transforms(kg, transforms=transforms)
    
    with open(cache_file, "wb") as f:
        pickle.dump(kg, f)
    print(f"💾 Cached {stage_name}")
    
    return kg



def create_knowledge_graph(generator_llm, generator_embeddings, page_generator):
    print("🔨 Building knowledge graph from scratch...")
    kg = KnowledgeGraph()
    
    # EXTRACTION: get information from each node, that is relevant to producing some relationship between them
    # TODO (optional): write custom extractors, for our domain
    # extractors take elements from each node (some specific elements)
    # and based on this establish properties of graph nodes

    for doc in page_generator:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc["document_chunk_content"], "document_metadata": doc["metadata"]}
            )
        )

    # the amount of nodes, should be the same as the amount of pages
    print("Nodes: ", len(kg.nodes))
    
    # extracts keywords
    keyphrase_extractor = KeyphrasesExtractor(llm=generator_llm, property_name="keyphrases")
    
    # extracts named entities
    ner_extractor = NERExtractor(llm=generator_llm, property_name="entities")

    # extract summary, create embedding -> we use these for embedding relationship creation
    summary_extractor = SummaryExtractor(llm=generator_llm, property_name="summary")
    embedd_extractor = EmbeddingExtractor(property_name="summary_embedding", embed_property_name="summary", embedding_model=generator_embeddings)


    summary_embedding_rel_builder = SummaryCosineSimilarityBuilder(
        property_name="summary_embedding", 
        new_property_name="summary_cosine_similarity",
        threshold=0.9
    )

    overlap_rel_builder = OverlapScoreBuilder(
        property_name="entities",
        new_property_name="entities_overlap",
        threshold=0.1,
    )

    # build KG transformation after transformation
    kg = apply_and_cache_stage(kg, [summary_extractor], "stage1_split", force_rebuild=False)
    kg = apply_and_cache_stage(kg, [Parallel(keyphrase_extractor, ner_extractor)], "stage2_features", force_rebuild=False)
    kg = apply_and_cache_stage(kg, [embedd_extractor], "stage2_1_features", force_rebuild=False)
    kg = apply_and_cache_stage(kg, [overlap_rel_builder, summary_embedding_rel_builder], "stage3_relationships", force_rebuild=True)

    return kg