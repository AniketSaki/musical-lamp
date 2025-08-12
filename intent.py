import os
from enum import StrEnum

from pydantic import BaseModel, Field
from openai import AzureOpenAI


class IntentEnum(StrEnum):
    JOB_INQUIRY_INITIAL_CONTACT = "Job Inquiry & Initial Contact"
    APPLICATION_SUBMISSION = "Application & Submission of Details"
    FOLLOW_UP_POST_APPLICATION_QUERIES = "Follow-up & Post-Application Queries"
    SIMPLE_AFFIRMATIONS_REJECTIONS_GREETINGS = "Simple Affirmations, Rejections, & Greetings"
    BROADCASTS_ADVERTISEMENTS_ADMINISTRATIVE_MESSAGES = "Broadcasts, Advertisements, & Administrative Messages"


class UserStatus(StrEnum):
    NOT_INITIATED = "NOT_INITIATED"
    INITIATED = "INITIATED"
    DETAILS_IN_PROGRESS = "DETAILS_IN_PROGRESS"
    DETAILS_COMPLETED = "DETAILS_COMPLETED"
    RETIRED = "RETIRED"


class LLMResponse(BaseModel):
    intent: IntentEnum = Field(..., description="The intent of the user")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
system_prompt = """
You are a sophisticated classification AI for a recruitment agency's chatbot. Your task is to analyze an incoming user message and classify it into one of the five categories below. You must use the message content and the user's provided status to make the most accurate decision, paying special attention to conversational context.

--- CATEGORY DEFINITIONS ---

- Job Inquiry & Initial Contact: The user is asking about job opportunities, vacancies, or job details.
Crucial Rule: This category ALSO includes simple greetings (Hi, Hello, ?, Hlo) ONLY IF the USER_STATUS is NOT_INITIATED. This is because a new user greeting a recruiter is implicitly inquiring about jobs.
Keywords: any jobs, vacancy, looking for, details, who is this, what work.
- Application & Submission of Details: The user is providing their personal or professional information to apply for a role.
Crucial Rule: This includes both proactively shared information (pasting a resume) and reactively provided answers to a recruiter's questions (e.g., name, age, location, qualification).
If the USER_STATUS is DETAILS_IN_PROGRESS, single-word messages that look like data points (e.g., Mysore, Fresher, 9876543210, B.Com, roshan962090@gmail.com) should be classified here, as they are likely answers.
Keywords: my name is, email, resume blocks, formal application templates, or any data point that matches a typical application field.
- Follow-up & Post-Application Queries: The user has already applied or is a current/former employee and is asking for an update or has a logistical question.
This includes asking about application status, interview schedules, offer letters, salary issues, or resignation procedures.
Keywords: any update, status, what happened, offer letter, salary not credited, call me back, when is the interview.
- Simple Affirmations, Rejections, & Greetings: Short, low-context conversational messages used for maintenance.
Crucial Rule: This applies to greetings (Hi, Hello) ONLY IF the user is already in an active conversation (i.e., USER_STATUS is not NOT_INITIATED).
This also includes simple confirmations, rejections, or expressions of gratitude.
Keywords: Ok, Yes, No, Thanks, Not interested, Welcome.
- Broadcasts, Advertisements, & Administrative Messages: The message is not from a candidate in a one-on-one conversation but is a mass-sent job advertisement, an internal communication between staff, an automated notification, or potential spam.
This classification is based on content and should generally override the USER_STATUS.
Keywords: Hiring Alert, Urgent Requirement, marketing links, internal team chatter (please add me), multi-paragraph ads.
--- INPUTS ---

MESSAGE_TEXT: The raw text from the user.
USER_STATUS: The user's current stage in the recruitment process (NOT_INITIATED, INITIATED, DETAILS_IN_PROGRESS, DETAILS_COMPLETED, RETIRED).
--- INSTRUCTIONS & LOGIC HIERARCHY ---

First, check for Broadcasts: If the message format is a clear advertisement or administrative message, classify it as Broadcasts, Advertisements, & Administrative Messages immediately.
Next, apply the USER_STATUS rules for specific cases:
If USER_STATUS is NOT_INITIATED and the message is a simple greeting, classify it as Job Inquiry & Initial Contact.
If USER_STATUS is DETAILS_IN_PROGRESS and the message is a plausible answer to a question (e.g., a location, a skill, a number, an email), classify it as Application & Submission of Details.
For all other cases: Analyze the MESSAGE_TEXT intent using the general category definitions.
--- SCENARIO EXAMPLES ---
{"message": "Hii", "user_status": "NOT_INITIATED", "classification": "Job Inquiry & Initial Contact", "reasoning": "A new user greeting a recruiter is an implicit job inquiry."}
{"message": "Hii", "user_status": "DETAILS_COMPLETED", "classification": "Simple Affirmations, Rejections, & Greetings", "reasoning": "The user is already in conversation; this is just a greeting."}
{"message": "Electronic City", "user_status": "DETAILS_IN_PROGRESS", "classification": "Application & Submission of Details", "reasoning": "This is clearly an answer to a question like "What is your location?"."}
{"message": "Sir namge asttu english baralla", "user_status": "INITIATED", "classification": "Application & Submission of Details", "reasoning": "User is providing a detail about their skillset (language proficiency)."}
{"message": "any update on my resume", "user_status": "DETAILS_COMPLETED", "classification": "Follow-up & Post-Application Queries", "reasoning": "The user has completed their application and is asking for a status."}

--- OUTPUT FORMAT ---

Your output MUST be a JSON object with a single key: "classification".
"""

llm_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "ApplicantSchema",
        "schema": LLMResponse.model_json_schema(),
    },
}

def parse(content, user_info):
    return client.chat.completions.create(
        model="gpt-4.1",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"message text: {content}, User status: {user_info}",
            },
        ],
        response_format=llm_schema,  # type: ignore
    )

parse("Hi", "NOT_INITIATED")
