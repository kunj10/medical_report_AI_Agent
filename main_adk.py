import os
from dotenv import load_dotenv 
from concurrent.futures import ThreadPoolExecutor
import asyncio 
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.agents import LlmAgent
from google.genai import types
from Utils.adk_agents import MedicalAgentFactory
import json

load_dotenv(dotenv_path='apikey.env') 

# read the medical report
medical_report_path = "Medical Reports/Medical Rerort - Michael Johnson - Panic Attack Disorder.txt"
if not os.path.exists(medical_report_path):
    print(f"Error: Medical report file not found at {medical_report_path}")
    exit()

with open(medical_report_path, "r") as file:
    medical_report = file.read()

# Initialize the Medical Agent Factory
agent_factory = MedicalAgentFactory()

# Create individual specialist agents using the factory
specialist_agents = {
    "Cardiologist": agent_factory.create_cardiologist_agent(medical_report),
    "Psychologist": agent_factory.create_psychologist_agent(medical_report),
    "Pulmonologist": agent_factory.create_pulmonologist_agent(medical_report)
}

# Asynchronous function to run each agent via a Runner instance
async def get_adk_agent_response(agent_name: str, agent: LlmAgent, input_content: str):
    print(f"{agent_name} is running...")
    try:
        session_service = InMemorySessionService()
        session = await session_service.create_session(
            app_name=f"{agent_name}_app",
            user_id="user_id",
            session_id=f"{agent_name}_session"
        )

        runner = Runner(
            agent=agent,
            app_name=f"{agent_name}_app",
            session_service=session_service
        )

        user_message_content = types.Content(
            role="user",
            parts=[types.Part(text=input_content)]
        )

        response_text = ""
        events = runner.run(
            user_id="user_id",
            session_id=session.id,
            new_message=user_message_content
        )

        for event in events:
            if event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            response_text += part.text
                break

        if not response_text:
             print(f"Warning: {agent_name} returned an empty response or did not have a final_response event with text content.")
        
        return agent_name, response_text
    except Exception as e:
        print(f"Error occurred with {agent_name}: {e}")
        return agent_name, None

# Main asynchronous function to orchestrate the agent runs
async def main():
    responses = {}
    
    tasks = [
        get_adk_agent_response(name, agent, medical_report)
        for name, agent in specialist_agents.items()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for agent_name, response_content in results:
        responses[agent_name] = response_content

    if not all(responses.values()):
        print("One or more specialist agents failed to generate a response. Cannot proceed with Multidisciplinary Team.")
        exit()

    team_agent = agent_factory.create_multidisciplinary_team_agent(
        cardiologist_report=responses["Cardiologist"],
        psychologist_report=responses["Psychologist"],
        pulmonologist_report=responses["Pulmonologist"]
    )

    print("MultidisciplinaryTeam is running...")
    try:
        team_session_service = InMemorySessionService()
        team_session = await team_session_service.create_session(
            app_name="MultidisciplinaryTeam_app",
            user_id="user_id",
            session_id="MultidisciplinaryTeam_session"
        )

        team_runner = Runner(
            agent=team_agent,
            app_name="MultidisciplinaryTeam_app",
            session_service=team_session_service
        )

        team_user_message_content = types.Content(
            role="user",
            parts=[types.Part(text="")]
        )

        final_diagnosis = ""
        team_events = team_runner.run(
            user_id="user_id",
            session_id=team_session.id,
            new_message=team_user_message_content
        )

        for event in team_events:
            if event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            final_diagnosis += part.text
                break

        if not final_diagnosis:
            print("Warning: MultidisciplinaryTeam returned an empty response or did not have a final_response event with text content.")

    except Exception as e:
        print(f"Error occurred with MultidisciplinaryTeam: {e}")
        final_diagnosis = "Error: Could not generate final diagnosis."

    final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis
    txt_output_path = "results/final_diagnosis.txt"

    os.makedirs(os.path.dirname(txt_output_path), exist_ok=True)

    with open(txt_output_path, "w") as txt_file:
        txt_file.write(final_diagnosis_text)

    print(f"Final diagnosis has been saved to {txt_output_path}")

if __name__ == "__main__":
    asyncio.run(main())