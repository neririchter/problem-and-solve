import operator
import re
import string
from collections import Counter
from typing import TypedDict, List, Annotated, Tuple, Union

import langfuse
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langfuse import Langfuse
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from sympy.integrals.meijerint_doc import category
from tavily import TavilyClient


import os

# get keys for your project from https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

# your openai key
# os.environ["OPENAI_API_KEY"] = ""

# Your host, defaults to https://cloud.langfuse.com
# For US data region, set to "https://us.cloud.langfuse.com"
os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

# from langfuse.callback import CallbackHandler
# langfuse_handler = CallbackHandler(
#     public_key="pk-lf-31ea01a1-7137-4ea5-9951-87b75dd8dc7d",
#     secret_key="sk-lf-94bedcae-69c6-4511-ae6f-36b650946d45",
#     host="http://localhost:3000"
# )

# init
langfuse = Langfuse(release = "v1")

def normalize_answer(s):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    if normalize_answer(prediction) == normalize_answer(ground_truth):
        return 1
    else:
        return 0

def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

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

class Plan(BaseModel):
    """Plan to follow in future"""

    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """For the given objective, come up with a simple step by step plan. \
This plan should involve individual tasks, that if executed correctly will yield the correct answer. Do not add any superfluous steps. \
The result of the final step should be the final answer. Make sure that each step has all the information needed - do not skip steps.""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner = planner_prompt | ChatOpenAI(
    model="gpt-4o", temperature=0, api_key=key
).with_structured_output(Plan)

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
    task = plan[0]
    task_formatted = f"""For the following plan:
{plan_str}\n\nYou are tasked with executing step {1}, {task}."""
    agent_response = agent_executor.invoke(
        {"messages": [("user", task_formatted)]}
    )
    return {
        "past_steps": [(task, agent_response["messages"][-1].content)],
    }


def plan_step(state: PlanExecute):
    plan = planner.invoke({"messages": [("user", state["input"])]})
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
    langfuse.create_dataset_item(
        dataset_name="dataset-questions",
        # any python object or value, optional
        input={
            "input": "what is the hometown of the mens 2024 Australia open winner?"
        },
        # any python object or value, optional
        expected_output={
            "response": "The hometown of Jannik Sinner, the winner of the men's 2024 Australian Open, is Sesto Pusteria (Sexten) in the northern part of Italy, in the county of Trentino Alto Adige, between the Alps."
        })

    return;

    langfuse.create_prompt(
        name="event-planner",
        prompt=
        "Plan an event titled {{Event Name}}. The event will be about: {{Event Description}}. "
        "The event will be held in {{Location}} on {{Date}}. "
        "Consider the following factors: audience, budget, venue, catering options, and entertainment. "
        "Provide a detailed plan including potential vendors and logistics.",
        config={
            "model": "gpt-3.5-turbo-1106",
            "temperature": 0,
        },
        labels=["production"]
    );
    return;
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
    app = workflow.compile().with_config({"run_name": "PlanAndExecute"})

    # config = {"recursion_limit": 50, "callbacks": [langfuse_handler]}

    dataset = langfuse.get_dataset("dataset-questions")



    for item in dataset.items:
        # Langchain callback handler that automatically links the execution trace to the dataset item

        handler = item.get_langchain_handler(run_name="Adapt")

        # Execute application and pass custom handler
        res = app.invoke(item.input, config = {"recursion_limit": 50, "callbacks": [handler]})

        # prv = None
        # for event in app.stream(inputs, config=config):
        #     for k, v in event.items():
        #         if k != "__end__":
        #             prv = v
        # print(prv)

        langfuse.score(
            id=handler.get_trace_id() + "correctness",
            trace_id=handler.get_trace_id(),
            name="correctness",
            value=1,  # 0 or 1
            data_type="BOOLEAN",  # required, numeric values without data type would be inferred as NUMERIC
            comment="Correct answer",  # optional
        )

        langfuse.score(
            id=handler.get_trace_id() + "f1",
            trace_id=handler.get_trace_id(),
            name="f1",
            value=f1_score(res['response'], item.expected_output['response']),  # 0 or 1
            data_type="NUMERIC",  # required, numeric values without data type would be inferred as NUMERIC
            comment="Correct answer",  # optional
        )

        langfuse.score(
            id=handler.get_trace_id() + "EM",
            trace_id=handler.get_trace_id(),
            name="EM",
            value=exact_match_score(res['response'], item.expected_output['response']),
            data_type="BOOLEAN",  # required, numeric values without data type would be inferred as NUMERIC
            comment="Correct answer",  # optional
        )


    # Flush the langfuse client to ensure all data is sent to the server at the end of the experiment run
    langfuse.flush()

    # inputs = {"input": "what is the hometown of the mens 2024 Australia open winner?"}
    # prv = None
    # for event in app.stream(inputs, config=config):
    #     for k, v in event.items():
    #         if k != "__end__":
    #             prv = v
    # print(prv)

    # chain.invoke(inputs, config={"callbacks": [langfuse_handler]})
if __name__ == "__main__":
    main()
