import json5 as json
from datetime import datetime
from typing import Optional, Dict

from langchain_community.adapters.openai import convert_openai_messages
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class SectionWriterInput(BaseModel):
    research_query: str = Field(description="The research query or topic for the section.")
    section_title: str = Field(description="The title of the specific section to write.")
    idx: int = Field(description="An index representing the order of this section (starting at 0")
    state: Optional[Dict] = Field(description="State of the research")


@tool("section_writer", args_schema=SectionWriterInput, return_direct=True)
async def section_writer(research_query, section_title, idx, state):
    """Writes a specific section of a research report based on the query, section title, and provided sources."""


    sources = state.get("sources").values()

    # Define the system and user prompts
    prompt = [{
        "role": "system",
        "content": (
            "You are an AI assistant that writes specific sections of research reports. "
            "You are provided with a research query, the section title, and a list of sources. "
            "Your task is to produce a detailed and structured section of the report."
        )
    }, {
        "role": "user",
        "content": (
            f"Today's date is {datetime.now().strftime('%d/%m/%Y')}.\n"
            f"Research Query: {research_query}\n"
            f"Section Title: {section_title}\n"
            f"Sources:\n{sources}\n\n"
            f"Write a JSON-formatted response where the section includes:\n"
            f"- 'title': The title of the section.\n"
            f"- 'content': A detailed and well-written content for the section.\n\n"
            f"Your response should be in the following format:\n"
            f"{sample_section}\n\nYour Section:"
        )
    }]

    # Convert prompts for OpenAI API
    lc_messages = convert_openai_messages(prompt)
    optional_params = {"response_format": {"type": "json_object"}}

    # Invoke OpenAI's model to get the section
    response = ChatOpenAI(model="gpt-4o-mini", max_retries=1, model_kwargs=optional_params).invoke(
        lc_messages
    ).content

    # Parse the JSON response and update the state
    section = json.loads(response)
    section['idx'] = idx
    state["sections"].append(section)

    tool_msg = f"Wrote the {section_title} Section, idx: {idx}"

    return state, tool_msg


sample_section = """
{
    "title": "Introduction to AI in the Workplace",
    "content": "This section explores the definition of AI and its applications in the workplace. AI technologies automate tasks, enhance productivity, and transform business operations.",
}
"""