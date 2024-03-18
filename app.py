from langchain import hub
from langchain.agents import load_tools, AgentExecutor, create_react_agent
from langchain_openai import AzureChatOpenAI
from streamlit_monaco import st_monaco
import streamlit as st
import os
import json
from langchain_core.agents import AgentActionMessageLog, AgentFinish
from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field

os.environ["AZURE_OPENAI_API_KEY"] = "AZURE_OPENAI_API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "AZURE_OPENAI_ENDPOINT"

class Response(BaseModel):
    """Final response to the question being asked"""

    answer: str = Field(description="The final answer to respond to the user")
    sources: List[int] = Field(
        description="List of page chunks that contain answer to the question. Only include a page chunk if it contains relevant information"
    )

def render():
    st.title("Graphql AI Force 🤖")
    st.write("Put you schema gql 👾")
    content = st_monaco(value="# Hello world", height="400px", language="graphql")
    st.write("I'm a graphql schema bot 🤖")
    prompt = st.text_input('Enter your query here:', key='prompt')
    submit_button = st.button('Submit', key='submit_button')

    if submit_button:
        llm = AzureChatOpenAI(
            openai_api_version="2023-12-01-preview",
            azure_deployment="t35t")

        tools = load_tools(
            ["graphql"],
            graphql_endpoint="https://search-svc-paris-cl.ecomm-stg.cencosud.com/graphql",
        )

        #llm_with_tools = llm.bind_functions([tools])

        prompt = hub.pull("hwchase17/react")

        #agent = (
        #    prompt
        #    | llm_with_tools
        #    | parse
        #)

        agent = create_react_agent(llm, tools, prompt)
        #agent = initialize_agent(
        #    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
        #)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            max_iterations=1
        )

        #result = agent.invoke(prompt + " stored in the graphql database that has this schema " + content)
        result = agent_executor.invoke({"input": prompt + " stored in the graphql database that has this schema " + content},
                                       return_only_outputs=True)
        st.write(result)

def parse(output):
    # If no function was invoked, return to user
    if "function_call" not in output.additional_kwargs:
        return AgentFinish(return_values={"output": output.content}, log=output.content)

    # Parse out the function call
    function_call = output.additional_kwargs["function_call"]
    name = function_call["name"]
    inputs = json.loads(function_call["arguments"])

    # If the Response function was invoked, return to the user with the function inputs
    if name == "Response":
        return AgentFinish(return_values=inputs, log=str(function_call))
    # Otherwise, return an agent action
    else:
        return AgentActionMessageLog(
            tool=name, tool_input=inputs, log="", message_log=[output]
        )

def main():
    render()

if __name__ == "__main__":
    main()
