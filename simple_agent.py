import os
from langchain.agents import AgentExecutor, AgentType, load_tools, initialize_agent
from langchain.chat_models import ChatAnyscale


def build_simple_agent():
  llm = ChatAnyscale(anyscale_api_base= "https://api.endpoints.anyscale.com/v1",
                     anyscale_api_key=os.getenv("ANYSCALE_API_KEY"),
                     model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
                     temperature=0.7,
                     verbose=True)
  
  tools = load_tools(tool_names=["llm-math", "dog-search"], llm=llm)
  agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
  
  print(agent.agent.llm_chain.prompt.template)
  return agent

def chat_with_agent(agent: AgentExecutor, user_input: str):
  output = agent.invoke({"input": user_input})
  return output

os.environ["ANYSCALE_API_KEY"] = "##################################"
agent = build_simple_agent()