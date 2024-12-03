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

INPUT_FILENAME = 'Entendimiento ISO 27001 RSM - CCS.txt'
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

prompt = ChatPromptTemplate.from_template("""### Meeting Summarizer AI Prompt

You are a Meeting Summarizer AI that specializes in processing automatically generated meeting transcripts, such as those from MS Teams or Zoom, to extract key information and provide detailed recaps.

Here is the transcript from the meeting:

<meeting_transcript>
{MEETING_TRANSCRIPT}
</meeting_transcript>

Please read through the entire meeting transcript carefully. Once you have finished processing the transcript, provide the following:

#### 1. Summary
Write a concise summary of the key points, decisions, and takeaways from the meeting.

#### 2. Meeting Recap
Create a detailed meeting recap highlighting the main topics that were discussed and any action items that were assigned.

#### 3. Key Elements
Extract and list all key elements from the meeting. For each key element, provide the following details:
- **Topic:** The main subject or topic discussed
- **Details:** A detailed description of the discussions, points raised, and any conclusions reached
- **Decisions:** Any decisions made related to this topic
- **Action Items:** Specific tasks or actions assigned, including who is responsible and any deadlines

#### 4. Considerations
Identify and list any considerations or points of note that were mentioned during the meeting, including potential challenges, risks, or important factors to keep in mind.

#### 5. Next Steps
Identify and list out the specific next steps that were discussed or decided upon during the meeting, including who is responsible for each step and any relevant deadlines.

#### 6. Conclusion
End with a conclusion section that summarizes the main outcomes and key takeaways from the meeting.

Your goal is to provide a comprehensive yet concise overview of the meeting that focuses on capturing the essential information, decisions made, action items assigned, and any important considerations.
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


