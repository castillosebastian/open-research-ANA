from importlib.resources import open_text
from pathlib import Path

from dotenv import load_dotenv
from typing import  Literal, cast
from datetime import datetime
import re
import json
import asyncio
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command
from langchain_core.runnables import RunnableConfig

from copilotkit.langgraph import copilotkit_emit_state, copilotkit_customize_config, copilotkit_exit, copilotkit_emit_message

from state import ResearchState
from config import Config
from tools.tavily_search import tavily_search
from tools.tavily_extract import tavily_extract
from tools.outline_writer import outline_writer
from tools.section_writer import section_writer

# Get absolute path to the .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path)

cfg = Config()

class MasterAgent:
    def __init__(self):
        self.tools = [tavily_search, tavily_extract, outline_writer, section_writer]
        self.tools_by_name = {tool.name: tool for tool in self.tools}

        # Define a graph
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("agent", self.call_model)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("human", self.ask_human)
        workflow.add_node("feedback", self.get_feedback)

        # Set the entrypoint as route_query
        workflow.set_entry_point("agent")

        workflow.add_edge("tools", "agent")
        workflow.add_edge("human", "feedback")
        workflow.add_edge("feedback", "agent")
        workflow.add_edge("agent", END)

        # Compile the graph, interrupt after human node to allow the frontend to invoke its tools.
        self.graph = workflow.compile(interrupt_after=['human'])

    # Define an async custom research tool node that access and updates the research state
    async def tool_node(self, state: ResearchState, config: RunnableConfig):
        # Add breakpoint here to inspect tool calls
        config = copilotkit_customize_config(config, emit_messages=False)
        msgs = []
        tool_state = {}
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name[tool_call["name"]]
            # Temporary messages struct that are accessible only to tools.
            state['messages'] = {'HumanMessage' if type(message) == HumanMessage else 'AIMessage' : message.content for message in state['messages']}
            tool_call["args"]["state"] = state  # update the state so the tool could access the state
            new_state, tool_msg = await tool.ainvoke(tool_call["args"])
            tool_call["args"]["state"] = None
            msgs.append(ToolMessage(content=tool_msg, name=tool_call["name"], tool_call_id=tool_call["id"]))
            tool_state = {
                "title": new_state.get("title", ""),
                "outline": new_state.get("outline", {}),
                "sections": new_state.get("sections", []),
                "sources": new_state.get("sources", {}),
                "proposal": new_state.get("proposal", {}),
                "logs": new_state.get("logs", []),
                "tool": new_state.get("tool", {}),
                "messages": msgs
            }
            await copilotkit_emit_state(config, tool_state)

        return tool_state


    @staticmethod
    def ask_human(state: ResearchState):
        # Define a fake node to ask human for feedback via frontend tools
        if cfg.DEBUG:
            print("**In human** waiting for human feedback from frontend")
        pass

    @staticmethod
    async def get_feedback(state: ResearchState, config: RunnableConfig):
        # Process human feedback from frontend
        config = copilotkit_customize_config(
            config,
            emit_messages=True,  # make sure to enable emitting messages to the frontend
        )
        last_message = cast(ToolMessage, state["messages"][-1])
        if cfg.DEBUG:
            print("**In get_feedback** received human feedback:\n",last_message)
        # Checking if the last frontend tool is 'review proposal' so we can update the proposal
        if last_message.name == 'review_proposal':
            reviewed_proposal = json.loads(last_message.content)
            if reviewed_proposal["approved"]:
                # Update outline with approved sections
                outline = {k: {'title': v['title'], 'description': v['description']} for k, v in
                           reviewed_proposal['sections'].items()
                           if isinstance(v, dict) and v.get('approved')}
                if cfg.DEBUG:
                    print("Setting outline: {}".format(outline))
                # await copilotkit_emit_message(config, "✅ Received approved outline! Moving forward with writing the sections")
                state['outline'] = outline
            # else:
            ## Can potentially emit messages to client
            #     await copilotkit_emit_message(config,
            #                                   "❌ Received rejected outline. Generating a new proposal based on your remarks.")
            # Update proposal
            state["proposal"] = reviewed_proposal

        return state

    async def call_model(self, state: ResearchState, config: RunnableConfig) -> Literal["tools", "human", "__end__"]:
        # Add breakpoint here to inspect the state before processing
        last_message = state['messages'][-1]
        if cfg.DEBUG:
             print(f"**In call_model**, last_message: {last_message}")
        allowed_types = (AIMessage, SystemMessage, HumanMessage, ToolMessage)

        if not isinstance(last_message, allowed_types):
            # Cast the last message to HumanMessage if it's of an unrecognized type
            last_message = HumanMessage(content=last_message.content)
            state['messages'][-1] = last_message

        outline = state.get("outline", {})
        sections = state.get("sections", [])
        proposal = state.get("proposal", {})

        prompt = (
                f"Today's date is {datetime.now().strftime('%d/%m/%Y')}.\n"
                "You are an expert research assistant, dedicated to helping users create comprehensive, well-sourced research reports. Your primary goal is to assist the user in producing a polished, professional report tailored to their needs.\n\n"
                "When writing a report use the following research tools:\n"
                "1. Use the tavily_search tool to start the research and gather additional information from credible online sources when needed.\n"
                "2. Use the tavily_extract tool to extract additional content from relevant URLs.\n"
                "3. Use the outline_writer tool to analyze the gathered information and organize it into a clear, logical **outline proposal**. Break the content into meaningful sections that will guide the report structure. You must use the outline_writer EVERY time you need to write an outline for the report\n"
                "4. After EVERY time you use the outline_writer tool, YOU MUST use review_proposal tool.\n"
                f"5. Once **outline proposal** is approved use the section_writer tool to write ONLY the sections of the report based on the **Approved Outline**{':' + str([outline[section]['title'] for section in outline]) if outline else ''} generated from the review_proposal tool. Ensure the report is well-written, properly sourced, and easy to understand. Avoid responding with the text of the report directly, always use the section_writer tool for the final product.\n\n"
                "After using the section_writer tool, actively engage with the user to discuss next steps. **Do not summarize your completed work**, as the user has full access to the research progress.\n"
                "Instead of sharing details like generated outlines or reports, simply confirm the task is ready and ask for feedback or next steps. For example:\n"
                "'I have completed [..MAX additional 5 words]. Would you like me to [..MAX additional 5 words]?'\n\n"
                "When you have a proposal, you must only write the sections that are approved. If a section is not approved, you must not write it."
                "Your role is to provide support, maintain clear communication, and ensure the final report aligns with the user's expectations.\n\n"
            )

        if proposal.get('remarks') and not outline:
            prompt += (
                f"**\nReviewed Proposal:**\n"
                f"Approved: {proposal['approved']}\n"
                f"Sections: {proposal['sections']}\n"
                f"User's feedback: {proposal['remarks']}"
                "You must use the outline_writer tool to create a new outline proposal that incorporates the user's feedback\n.")
        if outline:
                prompt += (
                    f"### Current State of the Report\n"
                    f"\n**Approved Outline**:\n{outline}\n\n"
                )
        report_content = ""
        for section in sections:
            report_content += (
            f"section {section['idx']} : {section['title']}\n"
            f"content : {section['content']}"
            f"footer : {section['footer']}\n\n")
        if report_content:
            prompt += f"**Report**:\n\n{report_content}"

        # if cfg.DEBUG:
        #     print("prompt: ", prompt[:3000])

        ainvoke_kwargs = {}
        ainvoke_kwargs["parallel_tool_calls"] = False

        # Get copilotkit actions with a default empty list if not present
        copilotkit_actions = state.get("copilotkit", {}).get("actions", [])

        response = await cfg.FACTUAL_LLM.bind_tools(
            self.tools + copilotkit_actions,
            **ainvoke_kwargs
        ).ainvoke([
            SystemMessage(content=prompt),
            *state["messages"],
        ], config)

        response = cast(AIMessage, response)

        # Use the same copilotkit_actions variable we defined above
        if response.tool_calls:
            for tool_call in response.tool_calls:
                # If the LLM makes a frontend tool call, route to the "human" node
                if any(action.get("name") == tool_call.get("name") for action in copilotkit_actions):
                    return Command(
                        goto="human",
                        update={
                            "messages": response
                        }
                    )
            # If the LLM makes a regular tool call, route to the "tools" node
            return Command(
                goto="tools",
                update={
                    "messages": response
                }
            )
        # If no conditions are met or if it's not an AIMessage, return END to stop
        return Command(
            goto=END,
            update={
                "messages": response
            }
        )

#Run the async function
    # used for running graph locally
    #Define an async function to run your graph code
    # async def run_graph(self):
    #     graph = self.graph
    #     initial_state = {
    #         "messages": [HumanMessage(content="Please run research on Tavily company")],
    #         "copilotkit": {"actions": []},  # Add required copilotkit structure
    #         "outline": {},
    #         "sections": [],
    #         "sources": {},
    #         "proposal": {},
    #         "logs": [],
    #         "tool": {}
    #     }
    #     async for s in graph.astream(initial_state, stream_mode="values"):
    #         message = s["messages"][-1]
    #         if isinstance(message, tuple):
    #             print(message)
    #         else:
    #             message.pretty_print()


#Run the async function
#asyncio.run(MasterAgent().run_graph())

graph = MasterAgent().graph
