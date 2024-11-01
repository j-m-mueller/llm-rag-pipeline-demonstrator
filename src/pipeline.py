from haystack.pipelines import Pipeline
from haystack.nodes import PromptNode, EmbeddingRetriever

class QueryPipeline:
    def __init__(self, retriever: EmbeddingRetriever, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the QueryPipeline with retriever and prompt node.

        :param retriever: The retriever component for document retrieval
        :param model_name: Name of the language model to use
        """
        self.retriever = retriever
        self.prompt_node = PromptNode(
            model_name_or_path=model_name,
            api_key="your-api-key",  # Replace with your actual API key
            default_prompt_template="Given the context, answer the question. Context: {join(documents)} Question: {query}"
        )
        
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
        if params is None:
            params = {"Retriever": {"top_k": 3}}
        
        return self.pipeline.run(query=query, params=params) 
