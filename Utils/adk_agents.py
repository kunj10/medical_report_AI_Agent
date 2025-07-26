
import os
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini 
from google.genai import types

class MedicalAgentFactory:
    def __init__(self):
        # Base instruction for all agents to maintain a professional and informative tone
        self.base_instruction = (
            "You are a highly specialized medical AI agent. "
            "Your responses should be professional, concise, and directly address the query based on provided medical information. "
            "Do not give medical advice or diagnoses directly; instead, summarize findings or analyses."
        )
        # Retrieve the API key from environment variables
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        if not self.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please ensure your .env file is configured correctly or set the environment variable."
            )

    def _create_base_agent(self, name: str, description: str, instruction_suffix: str) -> LlmAgent:
        """Helper to create a base LlmAgent with common configurations."""
        # Initialize the Gemini model with the API key
        llm_model = Gemini(
            model_id="gemini-1.5-flash",
            api_key=self.google_api_key
        )
        return LlmAgent(
            name=name,
            model=llm_model, # Pass the initialized Gemini model instance
            description=description,
            instruction=f"{self.base_instruction} {instruction_suffix}",
        )

    def create_cardiologist_agent(self, medical_report: str) -> LlmAgent:
        """Creates an LlmAgent specialized in cardiology."""
        instruction_suffix = (
            f"As a Cardiologist AI, analyze the following medical report focusing on cardiovascular aspects, "
            f"symptoms, test results, and potential implications for heart health. "
            f"Summarize your key findings and any concerns relevant to cardiology in a clear, structured format.\n\n"
            f"Medical Report:\n{medical_report}"
        )
        return self._create_base_agent(
            name="Cardiologist",
            description="Specializes in cardiovascular health and heart-related conditions.",
            instruction_suffix=instruction_suffix
        )

    def create_psychologist_agent(self, medical_report: str) -> LlmAgent:
        """Creates an LlmAgent specialized in psychology."""
        instruction_suffix = (
            f"As a Psychologist AI, analyze the following medical report focusing on mental health, "
            f"behavioral patterns, psychological symptoms, and emotional well-being. "
            f"Summarize your key findings and any concerns relevant to psychology, "
            f"especially regarding panic attack disorder, in a clear, structured format.\n\n"
            f"Medical Report:\n{medical_report}"
        )
        return self._create_base_agent(
            name="Psychologist",
            description="Specializes in mental health, psychological disorders, and emotional well-being.",
            instruction_suffix=instruction_suffix
        )

    def create_pulmonologist_agent(self, medical_report: str) -> LlmAgent:
        """Creates an LlmAgent specialized in pulmonology."""
        instruction_suffix = (
            f"As a Pulmonologist AI, analyze the following medical report focusing on respiratory health, "
            f"lung function, breathing difficulties, and any pulmonary conditions. "
            f"Summarize your key findings and any concerns relevant to pulmonology in a clear, structured format.\n\n"
            f"Medical Report:\n{medical_report}"
        )
        return self._create_base_agent(
            name="Pulmonologist",
            description="Specializes in respiratory system health and lung diseases.",
            instruction_suffix=instruction_suffix
        )

    def create_multidisciplinary_team_agent(self, cardiologist_report: str, psychologist_report: str, pulmonologist_report: str) -> LlmAgent:
        """
        Creates an LlmAgent representing a multidisciplinary team.
        It synthesizes reports from specialists to provide a comprehensive diagnosis.
        """
        combined_report = (
            f"Here are the reports from the specialist agents:\n\n"
            f"Cardiologist's Report:\n{cardiologist_report}\n\n"
            f"Psychologist's Report:\n{psychologist_report}\n\n"
            f"Pulmonologist's Report:\n{pulmonologist_report}\n\n"
        )
        instruction_suffix = (
            f"As the Multidisciplinary Medical Team AI, review the following specialist reports. "
            f"Synthesize this information to provide a comprehensive and holistic diagnosis for the patient. "
            f"Identify any correlations between the findings from different specialties. "
            f"Present the final diagnosis, including any recommended next steps or areas for further investigation, "
            f"in a structured and professional medical summary format.\n\n"
            f"{combined_report}"
        )
        return self._create_base_agent(
            name="MultidisciplinaryTeam",
            description="Synthesizes findings from multiple medical specialists to provide a holistic diagnosis.",
            instruction_suffix=instruction_suffix
        )