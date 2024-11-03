"""src.pipeline.pipeline.py -- Query pipeline implementation for document retrieval and LLM processing."""

import os

from haystack.nodes import EmbeddingRetriever, PromptNode
from haystack.pipelines import Pipeline


class QueryPipeline:
    def __init__(self, retriever: EmbeddingRetriever, model_name: str = "gpt-4o-mini"):
        """
        Initialize the QueryPipeline with retriever and prompt node.

        :param retriever: The retriever component for document retrieval
        :param model_name: Name of the language model to use
        """
        self.retriever = retriever
        
        PROMPT_TEMPLATE = """
        You are a helpful assistant. You need to provide answers to a QUESTION exclusively based on provided CONTEXT.
        
        CONTEXT: 
        
        {join(documents)}
        
        END OF CONTEXT.
        
        QUESTION:
        
        {query}
        
        END OF QUESTION.
        
        Make sure to limit your response exclusively to the provided CONTEXT. If the CONTEXT does not contain the answer,
        respond with "This question cannot be answered based on the given context."
        """
        
        self.prompt_node = PromptNode(
            model_name_or_path=model_name,
            api_key=os.environ.get("OPENAI_API_KEY"),
            default_prompt_template=PROMPT_TEMPLATE,
            max_length=500,  # maximum length for the prompt template
            model_kwargs={
                "temperature": 0.7  # randomness in response generation
            }
        )
        
        # set up pipeline
        self.pipeline = Pipeline()
        self.pipeline.add_node(component=self.retriever, name="Retriever", inputs=["Query"])
        self.pipeline.add_node(component=self.prompt_node, name="PromptNode", inputs=["Retriever"])
    
    def run(self, query: str, params: dict = None):
        """
        Run the pipeline with a query.

        :param query: The query string to process
        :param params: Optional parameters for the pipeline components
        :return: The pipeline results
        """
        params = {"Retriever": {"top_k": 5}} if params is None else params       
        
        result = self.pipeline.run(query=query, params=params)
        
        # ensure we have a consistent response format
        if "answers" not in result:
            result["answers"] = [{"answer": result.get("results", ["No answer generated."])[0]}]
        
        return result
    