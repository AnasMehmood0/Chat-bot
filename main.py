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
# The `__initial_auth_token` and `__firebase_config` are handled by the Canvas environment.
# For direct API calls within a Python script, ensure your environment variable is set.
# For this example, I will keep the provided placeholder.
gemini_api_key = "AIzaSyAnM_iTNVP_R1ilsIsmbjk9H_K_wMqNEfs"

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

@cl.on_chat_start
async def start():
    """
    This function is called when a new chat session starts.
    It initializes the AI model, agent, and sets up the chat history.
    It also sends the initial greeting message and sets up agent avatars.
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

    # --- UX Enhancement: Set a custom avatar for the agent ---
    # You need to place 'assistant_avatar.png' in a 'public' folder
    # at the root of your Chainlit project.
    await cl.Avatar(
        name="Anas's AI Assistant", # This name should match the agent's name
        url="/public/assistant_avatar.png", # Path to your image file
        size="large",
    ).send()

    # --- UX Enhancement: Define a more attractive and specific agent persona ---
    # This helps the LLM generate more consistent and engaging responses.
    agent: Agent = Agent(
        name="Anas's AI Assistant", # Name displayed in the UI
        instructions=(
            "You are a friendly, knowledgeable, and slightly enthusiastic AI assistant created by Anas. "
            "Your goal is to provide helpful information, answer questions clearly, and make interactions enjoyable. "
            "Always be polite, concise, and offer to assist further. "
            "You can use emojis sparingly to add a touch of warmth and personality. "
            "Do not act like a generic chatbot; strive to be engaging and proactive."
        ),
        model=model
    )
    cl.user_session.set("agent", agent) 
    
    # --- UX Enhancement: Send a more welcoming and engaging greeting ---
    # Includes action buttons for quick user interaction.
    await cl.Message(
        content=(
            "👋 Hey there! I'm Anas's AI Assistant, ready to help you out. "
            "What's on your mind today? Let's get started! 😊"
        ),
        actions=[ # Optional action buttons for quick user interaction
            cl.Action(name="suggest_topics", value="What can you do?", label="💡 Suggest topics"),
            cl.Action(name="tell_joke", value="Tell me a joke!", label="🤣 Tell me a joke"),
            cl.Action(name="give_feedback", value="I have feedback", label="✍️ Give feedback"),
        ]
    ).send()
        
@cl.on_message
async def main(message: cl.Message):
    """
    This function is called every time the user sends a message.
    It processes the message using the AI agent and sends back a response.
    """

    # Show a "thinking" message while the AI processes the request
    # --- UX Enhancement: Keep the animated thinking message ---
    msg = cl.Message(content="Thinking...✨✨🤔🤔")
    await msg.send()
    
    # Retrieve agent and config from user session
    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
    
    # Get current chat history and append the new user message
    history = cl.user_session.get("chat history") or []
    history.append({"role": "user", "content": message.content})
        
    try:
        # --- Debugging: Removed verbose print statements for cleaner console output ---
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
        
        # --- Debugging: Optional print statements for console logging ---
        print(f"User: {message.content}")
        print(f"Assistant: {response_content}")
            
    except Exception as e:
        # --- UX Enhancement: More user-friendly error message ---
        # Provide empathy and guidance instead of a raw error.
        msg.content = (
            f"Oops! Something went wrong on my end. 😥 I apologize for the inconvenience. "
            f"It seems I'm having trouble processing that request right now. "
            f"Please try again in a moment, or rephrase your request. "
            f"If the problem persists, feel free to give me feedback so Anas can take a look! "
            f"\n\n**(Error details for developer: `{str(e)}`)**" # Keep error details for you.
        )
        await msg.update()
        print(f"Error: {str(e)}") # Always log errors for debugging
