# MLinterviewer
A solo project using a multitude of libraries and APIs in order to (mostly) locally generate a realistic interview on the topic of your choosing
Follow the on-screen instructions to record and transcribe interviews.

# Project Overview
This project is designed to conduct interviews, record them, and generate transcriptions. It utilizes audio processing and machine learning technologies to create a seamless experience for recording interviews and converting them into text format.

# Features
Audio Recording: Leveraging PyAudio, the application can record interviews in high-quality audio.

Transcription: Utilizes advanced natural language processing (NLP) techniques to transcribe the recorded interviews accurately. (openai whisper v3)

Speech Synthesis: Converts text to speech for various purposes using state-of-the-art models. So far OpenAI's echo and Microsofts's latest local tts are available for selection at the start of the interview. Note that with longer responses from the interviewer, Microsoft's tts starts to fail catastrophically.


# Installation
Clone the repository:

git clone github.com/michaelmandich/MLinterviewer

Navigate to the project directory:

cd [project-directory]

Install required dependencies:

pip install -r requirements.txt

# Usage
Ensure you have the necessary API keys and environment variables set (for OpenAI, etc.).
Run the application:

python app.py

From here, the program will prompt you what to do in the UI, note that it must be nearly silent in your room for the recording to automatically end, and there is no way to manually end your recording yet. This is to come in a future update. A transcript will be generated and the recording saving function is still a WIP.

# Dependencies
PyAudio: For audio recording and processing.

PyDub: For handling audio files.

Transformers & Torch: For NLP tasks and speech synthesis.

OpenAI: For utilizing OpenAI's APIs (if used in the project).

Additional dependencies may be listed in requirements.txt.
# Configuration
Set the OPENAI_API_KEY in your environment variables for OpenAI functionalities.
Configure audio settings as per your requirements in the code.
# Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

License
Open-Source
