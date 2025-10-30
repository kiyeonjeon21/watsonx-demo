

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
    name ="user_flow_example_no_files",
    display_name="Quick User Survey",
    description="A simple user survey flow that demonstrates text, number input, and list display features.",
    input_schema=Name,
    output_schema=UserFlowResult
)
def build_user_flow(aflow: Flow = None) -> Flow:
    """
    Create a user flow demonstrating:
    - Welcome message with personalized greeting
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
        text="Hello {flow.input.first_name}! Welcome to our quick survey. Please answer a few questions."
    )

    # Step 2: Collect last name
    last_name_input = user_flow.field(
        direction="input",
        name="last_name", 
        display_name="Last Name", 
        kind=UserFieldKind.Text, 
        text="What's your last name?"
    )
    
    # Step 3: Collect age
    age_input = user_flow.field(
        direction="input",
        name="age", 
        display_name="Age", 
        kind=UserFieldKind.Number, 
        text="How old are you?"
    )

    # Step 4: Show available options (demonstrates List display)
    data_map = DataMap()
    data_map.add(Assignment(
        target_variable="self.input.value",
        value_expression="[\"Option A: Basic Plan\", \"Option B: Standard Plan\", \"Option C: Premium Plan\"]"
    ))
    options_display = user_flow.field(
        direction="output",
        name="subscription_options", 
        display_name="Recommended Plans", 
        kind=UserFieldKind.List, 
        text="Based on your profile, here are our recommended plans:",
        input_map=data_map
    )

    # Step 5: Thank you message
    thank_you_msg = user_flow.field(
        direction="output",
        name="thank_you", 
        display_name="Thank You", 
        kind=UserFieldKind.Text, 
        text="✓ Thank you for completing our survey, {flow.input.first_name}! Your information has been saved."
    )

    # Define the flow sequence
    user_flow.edge(START, welcome_msg)
    user_flow.edge(welcome_msg, last_name_input)
    user_flow.edge(last_name_input, age_input)
    user_flow.edge(age_input, options_display)
    user_flow.edge(options_display, thank_you_msg)
    user_flow.edge(thank_you_msg, END)
    
    # add the user flow to the flow sequence to create the flow edges
    aflow.sequence(START, user_flow, END)

    return aflow
