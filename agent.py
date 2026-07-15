from typing import Literal, TypedDict, Annotated
import operator
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode

PROMPT_TEMPLATE = """
Given this IoT data, evaluate if there is a risk of fires. 
If the data is missing critical context (like weather, historical patterns, or specific sensor baselines), 
mark yourself as unsure and request external data.
"""
llm = None  # Global variable to hold the LLM instance
app = None  # Global variable to hold the compiled graph app


# 1. Update structured output schema to allow uncertainty
class ThreatAssessment(BaseModel):
    danger: Literal["high", "medium", "low", "unsure"] = Field(
        description="Assessed danger level. Use 'unsure' if you lack sufficient data."
    )
    needs_external_data: bool = Field(
        description="Set to True if you need to query an external database, MCP server, or the internet."
    )
    search_query: str | None = Field(
        default=None,
        description="The query to search for if needs_external_data is True (e.g., 'current humidity in zone 4').",
    )


# 2. Update state to use Annotated for message appending
class AgentState(TypedDict):
    # Using Annotated and operator.add ensures messages are appended, not overwritten
    messages: Annotated[list, operator.add]
    assessment: dict


# 3. Define the LLM node logic
async def assess_danger(state: AgentState) -> dict:
    structured_llm = llm.with_structured_output(ThreatAssessment)

    # We pass the entire conversation history to the LLM so it has the context of previous searches
    prompt = (
        f"Analyze the data and determine the fire danger level:\n\n{state['messages']}"
    )

    result = await structured_llm.ainvoke(prompt)

    return {"assessment": result.model_dump()}


# 4. Define the external fetch node (Internet / MCP / DB)
# async def fetch_external_data(state: AgentState) -> dict:
#     query = state["assessment"].get("search_query", "general fire risk factors")
#     print(f"   [Tool] -> LLM is unsure. Fetching external data for: '{query}'...")

#     # TODO: Replace this mock with your actual MCP client call, DB query, or SerpAPI call
#     simulated_mcp_response = f"External System Data for '{query}': The local humidity dropped below 20% and wind speeds are 45km/h."

#     # Return the new information as a message to be appended to the state
#     return {"messages": [HumanMessage(content=simulated_mcp_response)]}


async def fetch_external_data(state: AgentState) -> dict:
    query = state["assessment"].get("search_query", "general fire risk factors")
    print(f"   [Tool] -> LLM is unsure. Fetching external data for: '{query}'...")

    try:
        # Initialize the async DuckDuckGo search client
        async with DDGS() as ddgs:
            # Fetch the top 3 text results asynchronously
            # (You can adjust max_results as needed for your context window)
            results = await ddgs.atext(query, max_results=3)

        if results:
            # Format the results into a readable string for the LLM
            formatted_results = "\n".join(
                [f"- {res.get('title', '')}: {res.get('body', '')}" for res in results]
            )
            search_response = (
                f"External Search Data for '{query}':\n{formatted_results}"
            )
        else:
            search_response = f"No external data found on DuckDuckGo for '{query}'."

    except Exception as e:
        # Fallback in case the search fails (e.g., rate limits or network issues)
        print(f"   [Tool Error] -> DuckDuckGo search failed: {e}")
        search_response = f"External search failed for '{query}'. Error: {str(e)}"

    # Return the new information as a message to be appended to the state
    return {"messages": [HumanMessage(content=search_response)]}


# 5. Define the routing logic
def route_assessment(state: AgentState) -> Literal["fetch_external_data", "__end__"]:
    assessment = state["assessment"]

    # If the LLM explicitly asked for data or marked danger as unsure, route to the tool
    if assessment.get("needs_external_data") or assessment.get("danger") == "unsure":
        return "fetch_external_data"

    # Otherwise, we have a confident assessment, so we end.
    return "__end__"


async def build_graph():
    mcp_client = MultiServerMCPClient(
        {
            # # Since the VPN doesn't allwo direct access to the ThingsBoard server to virtual machines, i deploy the mcp server locally.
            # "thingsboard": {
            #     "command": "java",
            #     "args": [
            #         "-jar",
            #         "thingsboard-mcp-server-2.1.0.jar",
            #         "--logging.level.root=ERROR",  # Muta i log informativi
            #         "--spring.main.banner-mode=off",  # Nasconde il banner testuale
            #     ],
            #     "env": {
            #         "THINGSBOARD_URL": "http://193.205.92.195:8080",
            #         "THINGSBOARD_USERNAME": "tenant@thingsboard.org",
            #         "THINGSBOARD_PASSWORD": "tenant",
            #         "THINGSBOARD_TOOLS_EDQ": "false",
            #         "THINGSBOARD_TOOLS_OTA": "false",
            #         "THINGSBOARD_TOOLS_GROUP": "false",
            #         "THINGSBOARD_TOOLS_USER": "false",
            #     },
            #     "transport": "stdio",
            # },
            # THINGSBOARD_URL="http://193.205.92.195:8080" THINGSBOARD_USERNAME="tenant@thingsboard.org" THINGSBOARD_PASSWORD="tenant" java -Dserver.port=8081 -Dspring.ai.mcp.server.stdio=false -Dspring.main.web-application-type=servlet -jar thingsboard-mcp-server-2.1.0.jar
            "thingsboard": {
                "url": "http://localhost:8081/sse",
                "transport": "sse",
            },
            "ddgs": {
                "command": "ddgs",
                "args": ["mcp"],
                "transport": "stdio",
            },
        }
    )
    LLM_MODEL = "gemini-2.5-flash-lite"
    LLM_API_KEY = "AQ.Ab8RN6I-GckFbpcdRhRNTKBG2_ZXfv_kJRcXbFC8A-Pq1o8CiA"
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

    global llm
    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=LLM_API_KEY,
        base_url=LLM_BASE_URL,
        temperature=0.2,  # Lower temperature is usually better for strict classification (da verificare)
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    tools = await mcp_client.get_tools()
    for tool in tools:
        print(f"   [Tool] -> Tool '{tool.name}' is available for use in the graph.")

    # 6. Construct graph routing
    workflow = StateGraph(AgentState)

    tool_node = ToolNode(tools)

    workflow.add_node("assess", assess_danger)
    # workflow.add_node("fetch_external_data", fetch_external_data)
    workflow.add_node("fetch_external_data", tool_node)

    workflow.add_edge(START, "assess")

    # Add the conditional routing
    workflow.add_conditional_edges("assess", route_assessment)

    # After fetching data, loop back to assess again with the new context
    workflow.add_edge("fetch_external_data", "assess")

    global app
    app = workflow.compile()


# 7. Execution function
async def call_agent(data):
    print("   [Agent] -> Avvio chiamata a LangGraph...")

    prompt_completo = f"{PROMPT_TEMPLATE}\n\n{data}"
    initial_state = {"messages": [HumanMessage(content=prompt_completo)]}

    try:
        result = await app.ainvoke(initial_state)
        print("   [Agent] -> Risposta finale ricevuta dall'LLM!")
        print(f"   [Agent] -> Valutazione: {result['assessment']}")

        return result["assessment"]

    except Exception as e:
        print(f"   [Agent] -> ERRORE FATALE dentro LangGraph: {e}")
        raise
