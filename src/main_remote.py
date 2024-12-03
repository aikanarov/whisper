import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional
import logging
from tqdm import tqdm
from pydub import AudioSegment
from openai import OpenAI
from src.env_config import env

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    """
    A class for transcribing audio files using OpenAI's Whisper model.

    This class handles the process of loading audio files, splitting them into
    manageable chunks, and transcribing them in parallel using the Whisper API.

    Parameters
    ----------
    input_path : str, optional
        Directory path for input audio files, by default 'data/input'
    output_path : str, optional
        Directory path for output transcription files, by default 'data/output'
    chunk_length_minutes : int, optional
        Length of each audio chunk in minutes, by default 10
    max_workers : int, optional
        Maximum number of concurrent workers for parallel processing, by default 3

    Attributes
    ----------
    input_path : Path
        Pathlib Path object for input directory
    output_path : Path
        Pathlib Path object for output directory
    chunk_length_ms : int
        Length of each audio chunk in milliseconds
    max_workers : int
        Maximum number of concurrent workers
    client : OpenAI
        OpenAI client instance for API calls
    """

    def __init__(
        self,
        input_path: str = 'data/input',
        output_path: str = 'data/output',
        chunk_length_minutes: int = 10,
        max_workers: int = 3
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.chunk_length_ms = chunk_length_minutes * 60 * 1000
        self.max_workers = max_workers
        self.client = OpenAI(api_key=env.openai_api_key.get_secret_value())
        
        # Ensure directories exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.input_path.mkdir(parents=True, exist_ok=True)

    def validate_audio_file(self, filepath: Path) -> bool:
        """
        Validate if the audio file exists and is in a supported format.

        Parameters
        ----------
        filepath : Path
            Path to the audio file to validate

        Returns
        -------
        bool
            True if the file is valid

        Raises
        ------
        FileNotFoundError
            If the audio file does not exist
        ValueError
            If the audio file format is not supported
        """
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        
        supported_formats = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
        if filepath.suffix.lower() not in supported_formats:
            raise ValueError(f"Unsupported audio format. Supported formats: {supported_formats}")
        
        return True

    def process_chunk(self, chunk: AudioSegment, chunk_index: int) -> str:
        """
        Process a single audio chunk and transcribe it.

        Parameters
        ----------
        chunk : AudioSegment
            Audio segment to process
        chunk_index : int
            Index of the chunk for identification

        Returns
        -------
        str
            Transcribed text from the audio chunk

        Raises
        ------
        Exception
            If there's an error during processing or transcription
        """
        temp_filename = f"temp_chunk_{chunk_index}_{int(time.time())}.mp3"
        temp_filepath = self.output_path / temp_filename

        try:
            # Export chunk to temporary file
            chunk.export(temp_filepath, format="mp3")
            
            # Transcribe the audio chunk
            with open(temp_filepath, "rb") as audio_file:
                result = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                return result.text

        except Exception as e:
            logger.error(f"Error processing chunk {chunk_index}: {str(e)}")
            raise
        finally:
            # Cleanup temporary file
            if temp_filepath.exists():
                temp_filepath.unlink()

    def transcribe_file(self, input_filename: str) -> Optional[str]:
        """
        Transcribe an audio file to text.

        This method handles the complete transcription process including:
        - Loading and validating the audio file
        - Splitting it into chunks
        - Processing chunks in parallel
        - Combining results
        - Saving the final transcription

        Parameters
        ----------
        input_filename : str
            Name of the input audio file in the input directory

        Returns
        -------
        Optional[str]
            Path to the output transcription file if successful, None otherwise

        Raises
        ------
        Exception
            If there's an error during any stage of the transcription process
        """
        input_filepath = self.input_path / input_filename
        output_filename = input_filepath.stem + ".txt"
        output_filepath = self.output_path / output_filename

        try:
            # Validate input file
            self.validate_audio_file(input_filepath)
            
            # Load the audio file
            logger.info(f"Loading audio file: {input_filename}")
            full_audio = AudioSegment.from_file(str(input_filepath))
            
            # Calculate chunks
            num_chunks = len(full_audio) // self.chunk_length_ms + (1 if len(full_audio) % self.chunk_length_ms > 0 else 0)
            chunks = [
                full_audio[i * self.chunk_length_ms:(i + 1) * self.chunk_length_ms]
                for i in range(num_chunks)
            ]
            
            # Process chunks in parallel
            transcriptions: List[str] = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_chunk = {
                    executor.submit(self.process_chunk, chunk, i): i
                    for i, chunk in enumerate(chunks)
                }
                
                # Process results as they complete
                with tqdm(total=len(chunks), desc="Transcribing") as pbar:
                    for future in as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            transcription = future.result()
                            transcriptions.append((chunk_idx, transcription))
                            pbar.update(1)
                        except Exception as e:
                            logger.error(f"Chunk {chunk_idx} failed: {str(e)}")
                            raise

            # Sort transcriptions by chunk index and combine
            transcriptions.sort(key=lambda x: x[0])
            full_transcription = "\n".join(text for _, text in transcriptions)
            
            # Save the result
            with open(output_filepath, 'w', encoding='utf-8') as file:
                file.write(full_transcription)
            
            logger.info(f"Transcription completed and saved to: {output_filepath}")
            return str(output_filepath)

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

def main():
    """
    Main entry point for the audio transcription program.

    This function initializes the AudioTranscriber and processes
    the default audio file. It includes basic error handling and
    logging of any errors that occur during execution.

    Raises
    ------
    Exception
        If there's an error during the transcription process
    """
    try:
        transcriber = AudioTranscriber()
        transcriber.transcribe_file('CAP 10 to MVP 01.m4a')
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise

if __name__ == "__main__":
    main()