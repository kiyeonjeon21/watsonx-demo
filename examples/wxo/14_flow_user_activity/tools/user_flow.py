

from typing import List
from pydantic import BaseModel, Field
from ibm_watsonx_orchestrate.flow_builder.flows import (
    Flow, flow, UserNode, START, END
)
from ibm_watsonx_orchestrate.flow_builder.types import Assignment, UserFieldKind
from ibm_watsonx_orchestrate.flow_builder.data_map import DataMap

class Name(BaseModel):
    """
    Input schema for the user flow.
    """
    first_name: str = Field(default="", description="Your first name")

class UserFlowResult(BaseModel):
    """
    Result of the user survey flow.
    
    Attributes:
        last_name (str): The person's last name.
        age (int): The person's age.
    """
    last_name: str = Field(description="Last name from user input")
    age: int = Field(description="Age from user input")

@flow(
    name ="user_flow_example",
    display_name="User Survey Form",
    description="A simple user survey flow that demonstrates various user interaction features.",
    input_schema=Name,
    output_schema=UserFlowResult
)
def build_user_flow(aflow: Flow = None) -> Flow:
    """
    Create a user flow demonstrating:
    - Welcome message with personalized greeting
    - File upload/download functionality  
    - Text input collection
    - Number input collection
    - List display
    """
    # user_flow which is a subflow to be added to the aflow
    user_flow = aflow.userflow()

    # Step 1: Welcome message (demonstrates dynamic text display)
    welcome_msg = user_flow.field(
        direction="output", 
        name="welcome", 
        display_name="Welcome", 
        kind=UserFieldKind.Text, 
        text="Hello {flow.input.first_name}! Welcome to the user survey demo. Let's collect some information."
    )
    
    # Step 2: File upload demonstration
    file_upload = user_flow.field(
        direction="input",
        name="upload", 
        display_name="Upload Document (Optional)", 
        kind=UserFieldKind.File,
        text="Please upload a document (PDF, DOC, etc.) if you have one."
    )

    # Step 3: Display uploaded file info
    file_display = user_flow.field(
        direction="output",
        name="file_status", 
        display_name="Upload Status", 
        kind=UserFieldKind.Text, 
        text="✓ Thank you for uploading your document."
    )

    # Step 4: File download demonstration
    file_download_map = DataMap()
    file_download_map.add(Assignment(
        target_variable="self.input.value",
        value_expression="flow[\"userflow_1\"][\"Upload Document (Optional)\"].output.value"
    ))
    file_download = user_flow.field(
        direction="output",
        name="download", 
        display_name="Download Processed File", 
        kind=UserFieldKind.File, 
        text="Here's your processed document:",
        input_map=file_download_map
    )

    # Step 5: Collect last name
    last_name_input = user_flow.field(
        direction="input",
        name="last_name", 
        display_name="Last Name", 
        kind=UserFieldKind.Text, 
        text="Please enter your last name:"
    )
    
    # Step 6: Collect age
    age_input = user_flow.field(
        direction="input",
        name="age", 
        display_name="Age", 
        kind=UserFieldKind.Number, 
        text="How old are you?"
    )

    # Step 7: Show available options (demonstrates List display)
    data_map = DataMap()
    data_map.add(Assignment(
        target_variable="self.input.value",
        value_expression="[\"Option A: Basic Plan\", \"Option B: Standard Plan\", \"Option C: Premium Plan\"]"
    ))
    options_display = user_flow.field(
        direction="output",
        name="subscription_options", 
        display_name="Available Plans", 
        kind=UserFieldKind.List, 
        text="Based on your profile, here are our recommended plans:",
        input_map=data_map
    )

    # Define the flow sequence
    user_flow.edge(START, welcome_msg)
    user_flow.edge(welcome_msg, file_upload)
    user_flow.edge(file_upload, file_display)
    user_flow.edge(file_display, file_download)
    user_flow.edge(file_download, last_name_input)
    user_flow.edge(last_name_input, age_input)
    user_flow.edge(age_input, options_display)
    user_flow.edge(options_display, END)
    
    # add the user flow to the flow sequence to create the flow edges
    aflow.sequence(START, user_flow, END)

    return aflow
