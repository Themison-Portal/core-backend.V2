from dotenv import load_dotenv
import asyncio
from pathlib import Path

from .knowledge_graph import create_knowledge_graph
from .personas import (persona_clinical_monitor,
                                        persona_participant_advocate,
                                        persona_principal_investigator, 
                                        persona_study_coordinator)

from .fetch_data_and_models import (fetch_data,
                                                    page_wise_window_generator,
                                                    get_pipeline_components)


from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer
)


from ragas.testset.synthesizers.multi_hop.abstract import (
    MultiHopAbstractQuerySynthesizer
)


from ragas.testset.synthesizers.single_hop.specific import (
        SingleHopSpecificQuerySynthesizer,
    )

from ragas.testset import TestsetGenerator



script_dir = Path(__file__).resolve().parent 
OUTPUT_DATASETS_PATH = script_dir / "data"
OUTPUT_DATASETS_PATH.mkdir(parents=True, exist_ok=True)


# Load environment variables from .env
load_dotenv()


async def generate_test_set(output_size: int, dataset_output_name: str, use_cache: bool = False):

    docs = await fetch_data()
    page_generator = page_wise_window_generator(docs)
    generator_embeddings, generator_llm = get_pipeline_components()

    kg = create_knowledge_graph(generator_llm, generator_embeddings, page_generator)

    print("Nodes: ", len(kg.nodes))
    print("Relationships: ", len(kg.relationships))

    
    personas = [persona_study_coordinator, persona_principal_investigator, persona_clinical_monitor, persona_participant_advocate]


    # for multihop we need to have relationships
    if (len(kg.relationships) == 0):
        raise Exception("Multi-hop requires at least one relationship")
    

    query_distribution = [
        (
            SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"),
            0.2,
        ),
        (
            SingleHopSpecificQuerySynthesizer(
                llm=generator_llm, property_name="entities"
            ),
            0.2,
        ),
        (
            MultiHopAbstractQuerySynthesizer(
            llm=generator_llm, 
            abstract_property_name="keyphrases",
            relation_property="summary_cosine_similarity"
            ),
            0.3,
        ),
        (
            MultiHopSpecificQuerySynthesizer(
                llm=generator_llm, 
                property_name="entities", 
                relation_type="entities_overlap",
                relation_overlap_property="overlapped_items"
            ), 
            0.3
        )
    ]

    # testset generation

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas,
    )
    # +1 on testset_size because for some reason first row is considered data
    testset = generator.generate(testset_size=output_size+1, query_distribution=query_distribution)
    df = testset.to_pandas()

    df.to_csv(OUTPUT_DATASETS_PATH / dataset_output_name)

    print(f"File saved successfully as {OUTPUT_DATASETS_PATH / dataset_output_name}")


async def main():
    await generate_test_set(output_size=100, dataset_output_name="test.csv", use_cache=True)


if __name__ == "__main__":
    asyncio.run(main())