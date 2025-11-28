the entry point of the backend is the main.py file.

all immediate end points are in the routes/routers and each router has its own purposes.

in the endpoints, you can find the dependecies for them and the most important one is the user check with the supabase authentication and the jwt returned by the frontend. so each call from the frontend is meant to be sent with the jwt token from supabase because right now, the frontend is communicating with the database directly. the second most important dependency is the get_db which returns the db session made by the sqlalchemy.

inside the endpoints, ou will find the services we provide to run the rag logics, the services will be found in the services folder which contain the interfaces, the implementations (indexing and retrieval) and the utils which store the functions under the hood forexample the prompts and llm calls.

in models you will find the python classes for the table generation schema which will be pushed automatically in the db dependency (it is commented out at the moment).

also it is important to note that although some of the steps are handled by langchain, we have a huggingface embedding provider which is extremely slow to generate embeddings, so we will have to resort to a vector embedding model like the openai text-embedding-3 small which is much faster.

In the supabase, there is an extension activated called pgvector which allows the creation of a column that takes in vector data.
