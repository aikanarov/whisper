import whisper
import os


PATH = '../data/'
INPUT_PATH = os.path.join(PATH, 'input')
OUTPUT_PATH = os.path.join(PATH, 'output')

INPUT_FILENAME = 'AI Cohort - Meeting.m4a'
INPUT_FILEPATH = os.path.join(INPUT_PATH, INPUT_FILENAME)

OUTPUT_FILENAME = os.path.splitext(INPUT_FILENAME)[0] + ".txt"
OUTPUT_FILEPATH = os.path.join(OUTPUT_PATH, OUTPUT_FILENAME)


model = whisper.load_model("large")
result = model.transcribe(INPUT_FILEPATH)

print(result["text"])

with open(OUTPUT_FILEPATH, 'w') as file:
    file.write(result["text"])
