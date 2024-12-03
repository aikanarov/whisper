# %% Import Libs

import os
from src.langchain_utils import get_llm_openai, get_llm_anthropic
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Model Name
DEFAULT_MODEL_NAME = 'gpt-4o'

# Paths
PATH = 'data/'
OUTPUT_PATH = os.path.join(PATH, 'output')

INPUT_FILENAME = 'Data room analysis using AI.txt'
INPUT_FILEPATH = os.path.join(OUTPUT_PATH, INPUT_FILENAME)

OUTPUT_FILENAME = os.path.splitext(INPUT_FILENAME)[0] + ".md"
OUTPUT_FILEPATH = os.path.join(OUTPUT_PATH, OUTPUT_FILENAME)

# %% Functions

def read_transcript_file(filepath):
    """
    Reads the content of a transcript file and returns it as a string.
    
    Args:
    - filepath (str): The path to the transcript file.
    
    Returns:
    - str: The content of the file.
    """
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    return content

meeting_transcript = read_transcript_file(INPUT_FILEPATH)


# Setup the Model based on the selected model name
model = get_llm_openai(DEFAULT_MODEL_NAME, max_tokens=4096, temperature=0.1)

prompt = ChatPromptTemplate.from_template("""You are a Meeting Summarizer AI that specializes in processing automatically generated meeting transcripts, such as those from MS Teams or Zoom, to extract key information and potential automation use cases.

Here is the transcript from the meeting about possible use cases to automate for the company:

<meeting_transcript>
{MEETING_TRANSCRIPT}
</meeting_transcript>

Please read through the entire meeting transcript carefully. Once you have finished processing the transcript, provide the following:

1. Write a concise summary of the key points, decisions, and takeaways from the meeting. 

2. Create a meeting recap highlighting the main topics that were discussed and any action items that were assigned.

3. List out all the potential use cases for automation that were mentioned during the meeting. For each use case, provide the following details:
- Use case: A brief name for the use case 
- Summary: A 1-2 sentence description of the use case
- Implementation: High-level thoughts on how this could be automated
- Data needed: What data would be required to automate this
- Complexity: An estimate of how complex it would be to automate (Simple, Moderate, Complex) 
- Additional info: Any other relevant notes about the use case

4. Identify and list out the specific next steps that were discussed or decided upon during the meeting.

5. End with a conclusion section that summarizes the main outcomes and key takeaways from the meeting.

Please structure your output exactly as specified above, with clear headings for each section. The use case details should follow the explicit format provided.

Your goal is to provide a comprehensive yet concise overview of the meeting that focuses on capturing the essential information, potential automation opportunities, and next steps.
""")

chain = prompt | model | StrOutputParser()

response = chain.invoke(
    {
        "MEETING_TRANSCRIPT": meeting_transcript,
        }
)

with open(OUTPUT_FILEPATH, 'w') as file:
    file.write(response)
print(f"Meeting report saved to {OUTPUT_FILEPATH}")


