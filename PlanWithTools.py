import json
import operator
import os
from typing import TypedDict, List, Annotated, Tuple, Union

from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from tavily import TavilyClient


os.environ['LANGCHAIN_TRACING_V2']='true'
os.environ['LANGCHAIN_ENDPOINT']="https://api.smith.langchain.com"
os.environ['LANGCHAIN_API_KEY']=""
os.environ['LANGCHAIN_PROJECT']="pr-complicated-injunction-39"

def search_tool(query: str) -> str:
    """
    Function to search for a query
    :param query: The search query
    :return: The search results
    """
    return tavily.get_search_context(query=query, search_depth="advanced")

Tools = [
                Tool(
                        name="SearchWeb",
                        func=search_tool,
                        description="""
                        Search the web for a query
                        """
                )
        ]

# Get the prompt to use - you can modify this!
prompt = hub.pull("ih/ih-react-agent-executor")

# tools = [search_tool]

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-4", api_key=key)
agent_executor = create_react_agent(llm, Tools, state_modifier=prompt)

class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str
    curStep: int

class Step(BaseModel):
    task: str = Field(description="Description of what needs to be done")
    toolName: str = Field(description="Name of the tool to use")
    toolArgs: str = Field(description="Input to provide to the tool")
    successCriteria: str =Field(description="The success criteria for the task")

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[Step] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, create a step-by-step plan where each step is either:
1. A manual task to be performed directly, or
2. A task requiring tool assistance

For tasks requiring tools, specify:
- Which tool to use from the available set
- The exact input/parameters needed for the tool
- What output to expect from the tool

Format each step as:
Step N: [Description of what needs to be done]
Tool required: [Tool name or "None"]
Input: [What to provide to the tool]
Expected output: [What the tool should return]

The final step's output should be the complete answer to the objective. Ensure each step:
- Contains all necessary information
- Builds logically on previous steps
- Includes any data transformations needed between tool outputs and next steps

Available Tools:
{tools}
""",
        ),
        ("placeholder", "{messages}"),
    ]
)
# planner_prompt.format_messages(tools= '{"name": "SearchWeb", "description": "Search the web for a query"}')

# print (planner_prompt)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0, api_key=key
).with_structured_output(Plan)

# print(planner.invoke(
#     {
#         "messages": [
#             ("user", "what is the hometown of the current Australia open winner?")
#         ],
#         "tools": '{"name": "SearchWeb", "description": "Search the web for a query"}'
#     }
# ))


class Response(BaseModel):
    """Response to user."""

    response: str


class Act(BaseModel):
    """Action to perform."""

    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )


replanner_prompt = ChatPromptTemplate.from_template(
    """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.

Your objective was this:
{input}

Your original plan was this:
{plan}

You have currently done the follow steps:
{past_steps}

Update your plan accordingly. If no more steps are needed and you can return to the user, then respond with that. Otherwise, fill out the plan. Only add steps to the plan that still NEED to be done. Do not return previously done steps as part of the plan."""
)


replanner = replanner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0, api_key=key
).with_structured_output(Act)

def execute_step(state: PlanExecute):
    plan = state["plan"]
    plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
    task = plan[state["curStep"]]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = agent_executor.invoke(
        {"messages": [("user", task_formatted)]
         }
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
        "curStep": state["curStep"] + 1
    }


def plan_step(state: PlanExecute):
    plan = planner.invoke(
                {
                    "messages": [
                        ("user", state["input"])
                    ],
                    "tools": '{"name": "SearchWeb", "description": "Search the web for a query"}'
                }
            )

        # {"messages": [("user", state["input"])],
        #                    "tools": '{"name": "SearchWeb", "description": "Search the web for a query"}'})
    return {"plan": plan.steps}


def replan_step(state: PlanExecute):
    output = replanner.invoke(state)
    if isinstance(output.action, Response):
        return {"response": output.action.response}
    else:
        return {"plan": output.action.steps}


def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "agent"

def main():
    workflow = StateGraph(PlanExecute)

    # Add the plan node
    workflow.add_node("planner", plan_step)

    # Add the execution step
    workflow.add_node("agent", execute_step)

    # Add a replan node
    workflow.add_node("replan", replan_step)

    workflow.add_edge(START, "planner")

    # From plan we go to agent
    workflow.add_edge("planner", "agent")

    # From agent, we replan
    workflow.add_edge("agent", "replan")

    workflow.add_conditional_edges(
        "replan",
        # Next, we pass in the function that will determine which node is called next.
        should_end,
        ["agent", END],
    )

    # Finally, we compile it!
    # This compiles it into a LangChain Runnable,
    # meaning you can use it as you would any other runnable
    app = workflow.compile().with_config({"run_name": "PlanWithTools"})

    config = {"recursion_limit": 50}
    inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}
    for event in app.stream(inputs, config=config):
        for k, v in event.items():
            if k != "__end__":
                print(v)

if __name__ == "__main__":
    main()
