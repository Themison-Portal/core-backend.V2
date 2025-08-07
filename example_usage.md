# Document ID-Based Querying

This feature allows you to query specific documents by their IDs instead of using semantic search. This is useful when you know exactly which documents you want to search through.

## API Usage

### Query with Document IDs

Send a POST request to `/query` with document IDs:

```json
{
  "message": "What are the key terms in the internship agreement?",
  "document_ids": ["uuid-1", "uuid-2", "uuid-3"],
  "retrieve_only": false
}
```

### Retrieve Documents Only (without generation)

```json
{
  "message": "Find relevant sections",
  "document_ids": ["uuid-1", "uuid-2"],
  "retrieve_only": true,
  "limit": 10
}
```

## How It Works

1. **Document ID Input**: The API accepts a list of document IDs in the request
2. **Chunk Retrieval**: The system queries the `document_chunks` table to get all chunks from the specified documents
3. **Content Aggregation**: All chunks from the specified documents are collected and formatted
4. **LLM Processing**: The aggregated content is sent to the LLM along with the user's query
5. **Streaming Response**: The LLM generates a response based on the specific document content

## Benefits

- **Precise Control**: Query only specific documents you're interested in
- **Faster Retrieval**: No need for semantic search when you know the target documents
- **Consistent Results**: Always get content from the same set of documents
- **Better Context**: LLM has access to complete document content rather than just semantically similar chunks

## Example Use Cases

1. **Legal Document Review**: Query specific contracts or agreements by their IDs
2. **Research Papers**: Search through a specific set of papers you've uploaded
3. **Technical Documentation**: Query specific manuals or guides
4. **Compliance Checking**: Review specific policy documents

## Response Format

The response includes metadata about which document and chunk each piece of content came from:

```json
{
  "content": "The internship agreement specifies...",
  "metadata": {
    "source": "Document uuid-1",
    "page": "Chunk 0",
    "chunk_id": "chunk-uuid",
    "document_id": "uuid-1"
  }
}
```

## Error Handling

- Invalid document IDs will result in empty results
- Missing document IDs will fall back to semantic search
- Database connection issues will return appropriate error messages
