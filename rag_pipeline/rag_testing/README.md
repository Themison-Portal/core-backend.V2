# Current issues
- when a question not related to the database is asked we get some response


# How to run:
In order to run synthetic dataset creation, run
```
python3 -m rag_pipeline.rag_testing.synthetic_dataset.synthetic_test_set
```
Remember to run it in the root of the project (pwd/core-backend.V2), and set (knowledge_graph)
```py
force_rebuild=True
```

In order to run ragas testing on the root of the project run:
```
python3 -m rag_pipeline.rag_testing.ragas
```
