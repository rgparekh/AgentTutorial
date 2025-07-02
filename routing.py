import os
import json
import requests
import logging

# import google.generativeai as genai
from google import genai
from google.genai import types
from google.genai.types import Tool
from pydantic import BaseModel, Field
from typing import Optional, Literal
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

class CalendarRequestType(BaseModel):
    """Router LLM call: Determine the type of calendar request"""

    request_type: Literal["new_event", "modify_event", "other"] = Field(
        description="Type of calendar request being made"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")
    description: str = Field(description="Cleaned description of the request")


class NewEventDetails(BaseModel):
    """Details for creating a new event"""

    name: str = Field(description="Name of the event")
    date: str = Field(description="Date and time of the event (ISO 8601)")
    duration_minutes: int = Field(description="Duration in minutes")
    participants: list[str] = Field(description="List of participants")


class Change(BaseModel):
    """Details for changing an existing event"""

    field: str = Field(description="Field to change")
    new_value: str = Field(description="New value for the field")


class ModifyEventDetails(BaseModel):
    """Details for modifying an existing event"""

    event_identifier: str = Field(
        description="Description to identify the existing event"
    )
    changes: list[Change] = Field(description="List of changes to make")
    participants_to_add: list[str] = Field(description="New participants to add")
    participants_to_remove: list[str] = Field(description="Participants to remove")


class CalendarResponse(BaseModel):
    """Final response format"""

    success: bool = Field(description="Whether the operation was successful")
    message: str = Field(description="User-friendly response message")
    calendar_link: Optional[str] = Field(description="Calendar link if applicable")

def run_model(model_name, contents, config):    
    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config=config
    )
    return response

# Define tools as functionsor modify
def route_calendar_request(user_input: str) -> CalendarRequestType:
    """Router LLM call to determine the type of calendar request"""
    logger.info("Routing calendar request")

    config = types.GenerateContentConfig(
        system_instruction = f"Determine if the request is to create a new calendar event or modify an existing one.",
        response_mime_type = "application/json",
        response_schema = CalendarRequestType
    )
    
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=user_input)]
        )
    ]

    response = run_model(model_name, contents, config)
    response_json = json.loads(response.candidates[0].content.parts[0].text) 

    logger.info(
        f"Extraction complete - Is calendar event: {response_json["request_type"]}, Confidence: {response_json["confidence_score"]:.2f}"
    )

    return response 

def handle_new_event(description: str) -> CalendarResponse:
    """Process a new event request"""
    logger.info("Processing new event request")

    config = types.GenerateContentConfig(
        system_instruction = f"Extract details for creating a new calendar event.",
        response_mime_type = "application/json",
        response_schema = NewEventDetails
    )
    
    contents = [
        types.Content(
            role="user", parts=[types.Part(text=description)]
        )
    ]

    response = run_model(model_name, contents, config)
    response_json = json.loads(response.candidates[0].content.parts[0].text) 

    logger.info(
        f"New calendar event: {response_json}"
    )

    # Generate response
    return CalendarResponse(
        success=True,
        message=f"New calendar event '{response_json["name"]}' created for {response_json["date"]} with {', '.join(response_json["participants"])}",
        calendar_link=f"calendar://new?event={response_json["name"]}",
    )

def handle_modify_event(description: str) -> CalendarResponse:
    """Process a modify event request"""
    logger.info("Processing modify event request")

    config = types.GenerateContentConfig(
        system_instruction = f"Extract details for modifying an existing calendar event.",
        response_mime_type = "application/json",
        response_schema = ModifyEventDetails
    )

    contents = [
        types.Content(
            role="user", parts=[types.Part(text=description)]
        )
    ]

    response = run_model(model_name, contents, config)
    response_json = json.loads(response.candidates[0].content.parts[0].text) 

    logger.info(
        f"Modify calendar event: {response_json}"
    )

    # Generate response
    return CalendarResponse(
        success=True,
        message=f"Modified calendar event '{response_json["event_identifier"]}'", 
        calendar_link=f"calendar://modify?event={response_json["event_identifier"]}",
    )

def process_calendar_request(user_input: str) -> Optional[CalendarResponse]:
    """Main function implementing the routing workflow"""
    logger.info("Processing calendar request")

    # Route the request
    route_result = route_calendar_request(user_input)

    result_json = json.loads(route_result.candidates[0].content.parts[0].text)

    logger.info(f"Route Result: {result_json}")

    # Check confidence threshold
    if result_json["confidence_score"] < 0.7:
        logger.warning(f"Low confidence score: {result_json["confidence_score"]}")
        return None

    # Route to appropriate handler
    if result_json["request_type"] == "new_event":
        return handle_new_event(result_json["description"])
    elif result_json["request_type"] == "modify_event":
        return handle_modify_event(result_json["description"])
    else:
        logger.warning("Request type not supported")
        return None


# --------------------------------------------------------------
# Step 3: Test with new event
# --------------------------------------------------------------

new_event_input = "Let's schedule a team meeting next Tuesday at 2pm with Alice and Bob"
result = process_calendar_request(new_event_input)
if result:
    print(f"Response: {result.message}")

# --------------------------------------------------------------
# Step 4: Test with modify event
# --------------------------------------------------------------

modify_event_input = (
    "Can you move the team meeting with Alice and Bob to Wednesday at 3pm instead?"
)
result = process_calendar_request(modify_event_input)
if result:
    print(f"Response: {result.message}")

# --------------------------------------------------------------
# Step 5: Test with invalid request
# --------------------------------------------------------------

invalid_input = "What's the weather like today?"
result = process_calendar_request(invalid_input)
if not result:
    print("Request not recognized as a calendar operation")