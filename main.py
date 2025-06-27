import os
from dotenv import load_dotenv
from typing import cast
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
import chainlit as cl

# Load environment variables from .env file
load_dotenv()

# --- IMPORTANT: API Key Handling ---
# In a real application, you should load this from environment variables
# or a secure configuration system, not hardcode it directly in the script.
gemini_api_key = "GEMINI_API_KEY" 
gemini_api_key = os.getenv(gemini_api_key)

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

@cl.on_chat_start
async def start():
    """
    This function is called when a new chat session starts.
    It initializes the AI model, agent, and sets up the chat history.
    It also sends the initial greeting message.
    """
    
    # Initialize the external OpenAI client for Gemini API
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/",
    )
    
    # Configure the Gemini-2.0-flash model
    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client,
    )
    
    # Set up the run configuration for the agent
    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True, # Disable tracing for cleaner output, enable for debugging
    )
    
    # Store chat history and config in user session
    cl.user_session.set("chat history", []) 
    cl.user_session.set("config", config)

    # Reverted to original, simpler agent instructions
    agent: Agent = Agent(
        name="Assistant", # Original name
        instructions="A helpful assistant that can answer questions and provide information.", # Original instructions
        model=model,
        tools=[] # Kept this line as it fixed a previous error.
    )
    cl.user_session.set("agent", agent) 
    
    # Reverted to original, simpler greeting message
    await cl.Message(content="Hello! I am your assistant by Anas 🤷‍♂. How can I help you today?").send()
        
@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time the user sends a message.
    It processes the message using the AI agent and sends back a response.
    """

    # Show a "thinking" message while the AI processes the request
    msg = cl.Message(content="Thinking...✨✨🤔🤔")
    await msg.send()
    
    # Retrieve agent and config from user session
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
    
    # Get current chat history and append the new user message
    history = cl.user_session.get("chat history") or []
    history.append({"role": "user", "content": message.content})
        
    try:
        # Debugging print statements are now optional and can be re-enabled if needed
        # print("\n[CALLing_AGENT_WITH_CONTEXT]\n", history, "\n[CALLing_AGENT_WITH_CONTEXT]\n")
            
        # Run the agent with the current chat history as input
        result = Runner.run_sync(
            starting_agent=agent,
            input=history,
            run_config=config,
        )
            
        # Extract the final output from the agent's run result
        response_content = result.final_output
        
        # Update the thinking message with the actual response content
        msg.content = response_content
        await msg.update()
            
        # Update chat history in the user session with the agent's complete exchange
        cl.user_session.set("chat history", result.to_input_list())
        
        # Debugging print statements for console logging
        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")
            
    except Exception as e:
        # Reverted to original, simpler error message
        msg.content = f"An error occurred: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
