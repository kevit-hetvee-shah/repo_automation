from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
# callback = UsageMetadataCallbackHandler()

class Workflow:
    def __init__(self, state, recursion_depth: int = 20, query: str = ""):
        self.state = state
        self.recursion_depth = recursion_depth
        self.query = query
        self.builder = None
        self.compiled_graph = None

    def generate_graph(self):
        builder = StateGraph(self.state)
        self.builder = builder
        return builder

    def generate_compiled_graph(self):
        compiled_graph = self.builder.compile()
        self.compiled_graph = compiled_graph
        return compiled_graph

    def invoke(self):
        result = self.compiled_graph.invoke({
            "messages": [
                HumanMessage(role="user", content=self.query)
            ],
        },
            config=RunnableConfig(recursion_limit=self.recursion_depth))
        return result

    def stream(self, query):
        result = self.compiled_graph.stream(
            {
                "messages": [
                    HumanMessage(role="user", content=query)
                ],
            },
            config=RunnableConfig(recursion_limit=self.recursion_depth)


        )
        for i in result:
            yield i
