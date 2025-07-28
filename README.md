# AI Agent

Issues:

1. Embedding model is not efficient (google-gemini)
2. Unable to upload massive chunks to pinecone. (PineconeBadRequestError: Vector dimension 0 does not match the dimension of the index 3072)
3. The responses are not quite efficient, may be due to LLM (google-gemini) & Very less vector database storage (~300 records)
4. Implement better chat memory for optimize query based on previous messages.
5. Unable to read xlsx files by default in LangChain. Need to use other tools or convert to csv

Improvements:

1. Each user records should be only able to queried by them. (Multi tenant achitecture)
2. Chat session should be stored in redis, whenever user back to chat need to upload chat session from redis to stateful.
3. It will be a better approach if using langgraph & mcp framework.
4. Using tools or different context may increase the summarization accuracy.
5. Use of frameworks like langsmith will be better to evaluate.
