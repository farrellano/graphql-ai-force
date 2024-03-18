from langchain import hub
from langchain.agents import load_tools, AgentExecutor, create_json_chat_agent
from langchain_openai import AzureChatOpenAI
from streamlit_monaco import st_monaco
import streamlit as st
import os

os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""


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

        toolsg = load_tools(
            ["graphql"],
            graphql_endpoint="https://search-svc-paris-cl.ecomm-stg.cencosud.com/graphql",
        )

        prompthub = hub.pull("hwchase17/react-chat-json")
        
        agent = create_json_chat_agent(llm, toolsg, prompthub)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=toolsg,
            verbose=True,
            max_iterations=1,
            handle_parsing_errors=True
        )
        result = agent_executor.invoke({"input": prompt + " stored in the graphql database that has this schema " + content},
                                       return_only_outputs=True)
        st.write(result)


def main():
    render()

if __name__ == "__main__":
    main()
