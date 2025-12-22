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
                                    EmbeddingExtractor
                                    )

from ragas.testset.transforms.splitters import (HeadlineSplitter)

from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer
)

from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer
)

from ragas.testset.synthesizers.multi_hop.base import (
    MultiHopQuerySynthesizer,
    MultiHopScenario,
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

async def fetch_data():

    # fetch data and sort it by page and creation, so 
    # we have the whole document in proper page-wise chunks
    async with AsyncSessionLocal() as session:
        sql = text(
            """
            SELECT dd.id, dd.document_id, dd.content, dd.page_number
            FROM document_chunks_docling dd
            ORDER BY dd.page_number ASC, dd.created_at ASC
            """
            )
        
    
        result = await session.execute(
            sql
        )

        return result.fetchall()


def page_wise_window_generator(docs):
    """
    Takes each row from the database, collects it into one page and yields a result
    for knowledge graph node's creation.
    
    :param chunks: Description
    """

    i = 0
    current_page = 1
    window = {"document_chunk_content": "",
            "metadata": {
                "document_id": "",
                "page_number": ""
            }
    }
    while i < len(docs):
        # we gather all chunks and yield each chunk
        row = docs[i]
        page = window
        page["metadata"]["document_id"] = str(row.document_id)
        page["metadata"]["page_number"] = str(current_page)
        while i < len(docs) and docs[i].page_number == current_page:
            page["document_chunk_content"] = row.content
            i += 1

        yield page

        # after processing one page go to the next one
        current_page += 1



def get_pipeline_components():
    # LLM
    chat_model = ChatOpenAI(model=LLM_MODEL_NAME, temperature=0.1)
    generator_llm = LangchainLLMWrapper(chat_model)
    openai_client = openai.OpenAI()

    # EMBEDDING
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    return generator_embeddings, generator_llm, openai_client




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


# @dataclass
# class MyMultiHopQuery(MultiHopQuerySynthesizer):

#     theme_persona_matching_prompt = ThemesPersonasMatchingPrompt()

#     async def _generate_scenarios(
#         self,
#         n: int,
#         knowledge_graph,
#         persona_list,
#         callbacks,
#     ) -> t.List[MultiHopScenario]:

#         # query and get (node_a, rel, node_b) to create multi-hop queries
#         results = kg.find_two_nodes_single_rel(
#             relationship_condition=lambda rel: (
#                 True if rel.type == "keyphrases_overlap" else False
#             )
#         )

#         num_sample_per_triplet = max(1, n // len(results))

#         scenarios = []
#         for triplet in results:
#             if len(scenarios) < n:
#                 node_a, node_b = triplet[0], triplet[-1]
#                 overlapped_keywords = triplet[1].properties["overlapped_items"]
#                 if overlapped_keywords:

#                     # match the keyword with a persona for query creation
#                     themes = list(dict(overlapped_keywords).keys())
#                     prompt_input = ThemesPersonasInput(
#                         themes=themes, personas=persona_list
#                     )
#                     persona_concepts = (
#                         await self.theme_persona_matching_prompt.generate(
#                             data=prompt_input, llm=self.llm, callbacks=callbacks
#                         )
#                     )

#                     overlapped_keywords = [list(item) for item in overlapped_keywords]

#                     # prepare and sample possible combinations
#                     base_scenarios = self.prepare_combinations(
#                         [node_a, node_b],
#                         overlapped_keywords,
#                         personas=persona_list,
#                         persona_item_mapping=persona_concepts.mapping,
#                         property_name="keyphrases",
#                     )

#                     # get number of required samples from this triplet
#                     base_scenarios = self.sample_diverse_combinations(
#                         base_scenarios, num_sample_per_triplet
#                     )

#                     scenarios.extend(base_scenarios)

#         return scenarios

# query = MyMultiHopQuery(llm=llm)
# scenarios = await query.generate_scenarios(
#     n=10, knowledge_graph=kg, persona_list=persona_list
# )




async def generate_test_set(output_size: int, dataset_output_name: str, use_cache: bool = False):
    docs = await fetch_data()

    page_generator = page_wise_window_generator(docs)

    generator_embeddings, generator_llm, openai_client = get_pipeline_components()


    # KNOWLEDGE GRAPH
    print("🔨 Building knowledge graph from scratch...")
    docs = await fetch_data()
    page_generator = page_wise_window_generator(docs)
    generator_embeddings, generator_llm, openai_client = get_pipeline_components()

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
    # extract embedding of the given page_content (needed for CosineSimilarityBuilder)
    embedd_extractor = EmbeddingExtractor(property_name="embedding", embedding_model=generator_embeddings)
    # extract headlines, to split nodes
    headline_extractor = HeadlinesExtractor(llm=generator_llm, property_name="headlines")

    # SPLITTING
    # to make the nodes more robust we introduce splitting mechanism, that
    # divides the nodes based on some criteria
    # min and max tokens are set to these values because the embedding 
    # for relationship creation uses different tokenizer
    splitter = HeadlineSplitter(
        min_tokens=150, 
        max_tokens=350
    )

    # build the graph using extracted data (if only extractors, there is no edges)
    # by default relationship building is sequential,
    # if we want to establish edges on independent builders we use Parallel
    # to establish edges we need to introduce a notion of similarity,

    # # we will build edges between jaccard similar objects (jaccard in terms of NER)
    # # the name of this relationship is jaccard_similarity
    # rel_builder = JaccardSimilarityBuilder(property_name="entities", new_property_name="jaccard_similarity")

    # TODO: requires writing a custom similarity query
    # cosine_rel_builder = SummaryCosineSimilarityBuilder(property_name= "summary_embedding",new_property_name= "summary_cosine_similarity")
    # jaccard_rel_builder = JaccardSimilarityBuilder(property_name="entities", new_property_name="jaccard_similarity")

    overlap_rel_builder = OverlapScoreBuilder(
        property_name="entities",
        new_property_name="entities_overlap",
        threshold=0.1,
    )

    # build KG transformation after transformation
    kg = apply_and_cache_stage(kg, [headline_extractor, splitter], "stage1_split", force_rebuild=False)
    kg = apply_and_cache_stage(kg, [Parallel(keyphrase_extractor, ner_extractor, embedd_extractor)], "stage2_features", force_rebuild=False)
    kg = apply_and_cache_stage(kg, [overlap_rel_builder], "stage3_relationships", force_rebuild=False)


    print("Nodes: ", len(kg.nodes))
    print("Relationships: ", len(kg.relationships))

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
    

    query_distribution = [
        (
            SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"),
            0.1,
        ),
        (
            SingleHopSpecificQuerySynthesizer(
                llm=generator_llm, property_name="entities"
            ),
            0.3,
        ),
        # (
        #     MultiHopAbstractQuerySynthesizer(llm=generator_llm),
        #     0.2,
        # ),

        (
        MultiHopSpecificQuerySynthesizer(
            llm=generator_llm, 
            property_name="entities", 
            relation_type="entities_overlap",
            relation_overlap_property="overlapped_items"
        ), 
        0.6
    ),

    ]

    # testset generation

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )

    testset = generator.generate(testset_size=output_size, query_distribution=query_distribution)
    df = testset.to_pandas()

    df.to_csv(OUTPUT_DATASETS_PATH / dataset_output_name)

    print(f"File saved successfully as {OUTPUT_DATASETS_PATH / dataset_output_name}")


async def main():
    await generate_test_set(output_size=20, dataset_output_name="test.csv", use_cache=True)


if __name__ == "__main__":
    asyncio.run(main())