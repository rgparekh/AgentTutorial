import os
import json
import requests
import logging

# import google.generativeai as genai
from google import genai
from google.genai import types
from google.genai.types import Tool
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
model_name = "gemini-2.0-flash"

# --------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------


class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""

    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
    """Second LLM call: Parse specific event details"""

    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )

def run_model(model_name, contents, config):    
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config
    )
    return response

# Define tools as functions
def extract_event_info(user_input: str) -> EventExtraction:
    """First LLM call to determine if input is a calendar event"""
    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    config = types.GenerateContentConfig(
        system_instruction = f"{date_context} Analyze if the text describes a calendar event.",
        response_mime_type = "application/json",
        response_schema = EventExtraction
    )
    
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=user_input)]
        )
    ]

    response = run_model(model_name, contents, config)
    response_json = json.loads(response.candidates[0].content.parts[0].text) 

    logger.info(
        f"Extraction complete - Is calendar event: {response_json['is_calendar_event']}, Confidence: {response_json['confidence_score']:.2f}"
    )

    return response 

def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to extract specific event details"""
    logger.info("Starting event details parsing")

    today = datetime.now()
    date_context = f"Today is {today.strftime('%A, %B %d, %Y')}."

    config = types.GenerateContentConfig(
        system_instruction = f"{date_context} Parse the following event description into a structured event object.",
        response_mime_type = "application/json",
        response_schema = EventDetails
    )
    
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=description)]
        )
    ]

    response = run_model(model_name, contents, config)
    response_json = json.loads(response.candidates[0].content.parts[0].text) 

    logger.info(
        f"Parsed event details - Name: {response_json['name']}, Date: {response_json['date']}, Duration: {response_json['duration_minutes']}"
    )
    logger.debug(f"Participants: {', '.join(response_json['participants'])}")

    return response 

def generate_confirmation(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate a confirmation message"""
    logger.info("Generating confirmation message")

    config = types.GenerateContentConfig(
        system_instruction = f"Generate a natural confirmation message for the event. Sign of with your name; Susie",
        response_mime_type = "application/json",
        response_schema = EventConfirmation
    )
    
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=event_details.model_dump_json())]
        )
    ]

    response = run_model(model_name, contents, config)
    response_json = json.loads(response.candidates[0].content.parts[0].text) 

    logger.info(f"Confirmation message generated: {response_json['confirmation_message']}")

    return response

def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function implementing the prompt chain with gate check"""
    logger.info("Processing calendar request")
    logger.debug(f"Raw input: {user_input}")

    # First LLM call: Extract basic info
    response = extract_event_info(user_input)
    response_json = json.loads(response.candidates[0].content.parts[0].text) 

    # Gate check: Verify if it's a calendar event with sufficient confidence
    if (
        not response_json["is_calendar_event"]
        or response_json["confidence_score"] < 0.7
    ):
        logger.warning(
            f"Gate check failed - is_calendar_event: {response_json['is_calendar_event']}, confidence: {response_json['confidence_score']:.2f}"
        )
        return None

    logger.info("Gate check passed, proceeding with event processing")

    # Second LLM call: Get detailed event information
    event_details = parse_event_details(response_json["description"])

    # Third LLM call: Generate confirmation
    confirmation = generate_confirmation(event_details)

    logger.info("Calendar request processing completed successfully")
    return confirmation


# --------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------

user_input = "Dentist's appointment next Friday from 8:30 AM to 10:00 AM PT. Leave at least 30 minutes before the appointment."

result = process_calendar_request(user_input)
if result:
    result_json = json.loads(result.candidates[0].content.parts[0].text) 
    print(f"Confirmation: {result_json['confirmation_message']}")
    if result_json["calendar_link"] is not None:
        print(f"Calendar Link: {result_json['calendar_link']}")
else:
    print(f"Request: '{user_input}' doesn't appear to be a calendar event request.")


# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

user_input = "Can you send an email to Alice and Bob to discuss the project roadmap?"

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print(f"Request: '{user_input}' doesn't appear to be a calendar event request.")
