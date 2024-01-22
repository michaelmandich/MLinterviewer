import json
import numpy as np
import openai
import os
import soundfile as sf
import pyaudio
import threading
import time
import tkinter as tk
import torch
import wave
from datasets import load_dataset
from pathlib import Path
from playsound import playsound
from pydub import AudioSegment
from queue import Queue
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import threading
import queue
from transformers import T5Tokenizer
from PIL import ImageTk, Image

# Constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
SILENCE_THRESHOLD = 2000
RECORD_SECONDS = 3
FILENAME = "recording.mp3"
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
openai.api_key = os.getenv('OPENAI_API_KEY')
client = openai.OpenAI(api_key=openai.api_key)
gui_queue = queue.Queue()


# Initialize PyAudio for dubbing
p = pyaudio.PyAudio()
message_queue = Queue()

# initialize voice synth for microsoft
synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

def wait_on_run(run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        time.sleep(0.5)
    return run

def play_audio(file_path):
    playsound(file_path, block = True)

def create_thread():
    thread = client.beta.threads.create()
    return thread



def send_message_and_get_response(client, message_content, thread_num, assistant_id, conversation):
    # Create a message in the thread
    client.beta.threads.messages.create(
        thread_id=thread_num,
        role="user",
        content=" " + str(message_content)
    )

    # Create a run
    run = client.beta.threads.runs.create(
        thread_id=thread_num,
        assistant_id=assistant_id,
    )

    # Wait for the run to complete
    run = wait_on_run(run, conversation)

    # Retrieve all messages in the thread
    messages_response = client.beta.threads.messages.list(thread_id=thread_num)

    # Get the last assistant message
    response = get_last_assistant_message(messages_response)
    generate_transcript_to_file(messages_response,"/home/michael/Desktop/first/transcript.txt")
    return response

# Function to check silence
def is_silent(data, threshold):
    return np.average(np.abs(np.frombuffer(data, dtype=np.int16))) < threshold

# Function to handle audio recording
def record_audio():
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording started...")
    frames = []
    silent_for = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        if is_silent(data, SILENCE_THRESHOLD):
            silent_for += 1
        else:
            silent_for = 0

        if silent_for >= (RATE / CHUNK) * RECORD_SECONDS:
            break

    print("Finished recording.")
    stream.stop_stream()
    stream.close()
    return frames

# Function to save audio
def save_audio(frames, filename):
    audio_segment = AudioSegment(
        data=b''.join(frames),
        sample_width=p.get_sample_size(FORMAT),
        frame_rate=RATE,
        channels=CHANNELS
    )
    audio_segment.export(filename, format="mp3")

def get_last_assistant_message(messages_response):
    # Convert the cursor to a list if necessary
    if not isinstance(messages_response.data, list):
        messages = list(messages_response.data)
    else:
        messages = messages_response.data
    print(messages)

    # Initialize a variable to store the last assistant message
    last_assistant_message = ""

    # Iterate through messages to find the last assistant message
    for message in messages:
        if message.role == 'assistant':
            # Get the content of the assistant message
            for content in message.content:
                if hasattr(content, 'text'):
                    last_assistant_message = content.text.value

            break

    # Remove the Markdown code block delimiters and the 'python' keyword from the last assistant message
    if last_assistant_message:
        cleaned_content = last_assistant_message.replace('\n```', '').strip()
        return cleaned_content

    return ""

def make_interviewer_voiceclip(words_to_say, speaker_embedding=None, output_filename="voice.wav"):
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    max_length = 600

    tokens = tokenizer.tokenize(words_to_say)
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    words_to_say = tokenizer.convert_tokens_to_string(tokens)

    if speaker_embedding is None:
        # Load a default speaker embedding
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    # Generate speech
    speech = synthesiser(words_to_say, forward_params={"speaker_embeddings": speaker_embedding})

    # Write the speech to a file
    sf.write(output_filename, speech["audio"], samplerate=speech["sampling_rate"])
    return output_filename

def openai_tts(text):

    #Convert the provided text to speech using OpenAI's TTS and save to a file.
    #Args:
    #text (str): Text to convert to speech.
    #Returns:
    #str: Filename of the saved audio file.
    # Generate a safe filename from the text
    safe_filename = "".join(c for c in text[:15] if c.isalnum() or c in " _-").rstrip()
    audio_file_path = Path("/home/michael/Desktop/first/recording.mp3")

    # Create the speech audio from the text
    response = openai.audio.speech.create(
        model="tts-1-hd",
        voice="echo",
        input=text
    )

    # Stream the response to the output file
    response.stream_to_file(audio_file_path)

    # Return the filename of the saved audio file
    return str(audio_file_path)

def generate_transcript_to_file(messages_response, file_path):
    # Convert the cursor to a list if necessary
    if not isinstance(messages_response.data, list):
        messages = list(messages_response.data)
    else:
        messages = messages_response.data

    # Initialize a variable to store the full transcript
    full_transcript = ""

    # Iterate through messages to create the transcript in the correct order
    for message in reversed(messages):
        # Check the role of the message (user or assistant)
        role_prefix = "User: " if message.role == 'user' else "Assistant: "

        # Get the content of the message
        for content in message.content:
            if hasattr(content, 'text'):
                message_text = content.text.value
                full_transcript += role_prefix + message_text + "\n\n"

    # Save the transcript to a file
    with open(file_path, "w") as file:
        file.write(full_transcript)

    # Update the transcript_output widget
    transcript_output.config(state=tk.NORMAL)  # Enable editing
    transcript_output.delete(1.0, tk.END)  # Clear existing content
    transcript_output.insert(tk.END, full_transcript)  # Update with new transcript
    transcript_output.config(state=tk.DISABLED)  # Disable editing

    print(f"Transcript saved to {file_path}")

# Function to convert speech to text
def speech_to_text(filename):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(filename)
    return result["text"]

# Function to handle the entire recording process
def record():
    frames = record_audio()
    save_audio(frames, FILENAME)
    text = speech_to_text(FILENAME)
    print(text)


"""
 "alldone": "alldone.png",
    "finishing": "aok.png",
    "thinking": "justthinking2.png",
    "working": "justthinking.png",
    "listening": "listening.png",
    "talking": "mouth open.png"

"""
# Function to start recording in a new thread
def handle_interview_process():
    conversation_thread = create_thread()
    selected_voice_provider = voice_provider_var.get()
    update_message("Please start by saying name and topic, and to exit interview, say something about exiting to Reiman")
    root.update()
    time.sleep(5)
    # Display message about selected voice and wait for 3 seconds
    update_message(f"{selected_voice_provider.capitalize()} voice selected, starting in 3 seconds...")
    root.update()
    time.sleep(3)
    # Initial setup
    final_audio = AudioSegment.empty()
    first_iteration = True

    while True:
        update_message("recording...")
        update_avatar("listening")  #
        root.update()
        frames = record_audio()
        update_message("finished recording...")
        update_avatar("working")  # Set avatar to "working"
        root.update()
        save_audio(frames, "recording.mp3")
        update_message("transferring voice to text...")
        update_avatar("thinking")  # Set avatar to "thinking"
        root.update()
        message_to_assistant = speech_to_text("/home/michael/Desktop/first/recording.mp3")

        update_message("Getting message from assistant...")
        update_avatar("working")  # Set avatar to "listening"
        root.update()
        response_from_interviewer = send_message_and_get_response(client, message_to_assistant, conversation_thread.id, "asst_XD9PE5vkN9HeFxgKdiqWeaRW", conversation_thread)
        if(response_from_interviewer == "stop"):
            interviewee_voice = AudioSegment.from_file("recording.mp3")
            final_audio = final_audio + interviewee_voice
            break
        update_message("Making interviewer audio file...")
        update_avatar("thinking")  # Set avatar to "working"
        root.update()

        # Creating interviewer audio file
        if (selected_voice_provider == "microsoft"):
            interviewer_voice = make_interviewer_voiceclip(response_from_interviewer)
        if (selected_voice_provider == "openai"):
            interviewer_voice = openai_tts(response_from_interviewer)
        if not first_iteration:
            interviewee_voice = AudioSegment.from_file("recording.mp3")
            final_audio = final_audio + interviewee_voice
        else:
            first_iteration = False
        # Concatenate the interviewer's response
        final_audio += AudioSegment.from_file(interviewer_voice)
        # Skip the first "recording.mp3"
        update_message("playing audio...")
        update_avatar("talking")  # Set avatar to "alldone"
        root.update()
        play_audio(interviewer_voice)

    # Export the final concatenated audio file
    final_audio.export("final_interview.mp3", format="mp3")



def stop_interview():
    # Access the global flag
    global stop_interview_flag
    # Set the flag to True when the stop button is clicked
    stop_interview_flag = True

# Create the UI
def update_message(updatetext):
    message_output.config(state=tk.NORMAL)  # Enable editing
    message_output.insert(tk.END, updatetext + '\n')
    message_output.config(state=tk.DISABLED)  # Disable editing

def stop_interview():
    exit()

def start_interview():
    interview_thread = threading.Thread(target=handle_interview_process)
    interview_thread.start()

def check_queue():
    while not gui_queue.empty():
        updatetext = gui_queue.get()
        message_output.insert(tk.END, updatetext + '\n')
    root.after(100, check_queue)

# Import the necessary libraries
# Define the avatar image paths for each state
avatar_images = {
    "alldone": "alldone.png",
    "finishing": "aok.png",
    "thinking": "thinking.png",
    "working": "justthinking.png",
    "listening": "listening.png",
    "talking": "mouth open.png"
}

# Function to update the avatar
def update_avatar(state):
    # Get the image path based on the state
    image_path = avatar_images.get(state)
    
    # Load the image
    avatar_image = Image.open(image_path)
    
    # Resize the image to fit the desired size
    avatar_image = avatar_image.resize((200, 200), Image.LANCZOS)
    
    # Convert the image to Tkinter-compatible format
    avatar_photo = ImageTk.PhotoImage(avatar_image)
    
    # Update the avatar image
    avatar_label.config(image=avatar_photo)
    avatar_label.image = avatar_photo





root = tk.Tk()
root.title("Audio Recorder")
root.geometry("1300x300+100+100")  # Width x Height + X offset + Y offset

# Variable to hold the selected voice provider
voice_provider_var = tk.StringVar(value="openai")

# Radio buttons for voice provider selection
tk.Radiobutton(root, text="OpenAI", variable=voice_provider_var, value="openai").grid(row=0, column=0, padx=10, pady=10)
tk.Radiobutton(root, text="Microsoft", variable=voice_provider_var, value="microsoft").grid(row=0, column=1, padx=10, pady=10)

# Adding buttons
start_button = tk.Button(root, text="Start Interview", command=start_interview)
stop_button = tk.Button(root, text="Stop Interview", command=stop_interview)
start_button.grid(row=0, column=2, padx=5, pady=5)
stop_button.grid(row=0, column=3, padx=5, pady=5)

# Message output area
message_output = tk.Text(root, height=10, width=50)
message_output.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='ew')  # Spanned across 2 columns for alignment
message_output.config(state=tk.DISABLED)  # Disable editing

# Scrollbar for message output
scrollbar = tk.Scrollbar(root, command=message_output.yview)
scrollbar.grid(row=1, column=2, sticky='ns')  # Adjusted sticky to 'ns' for vertical alignment
message_output['yscrollcommand'] = scrollbar.set

# Create a Text widget for the transcript
transcript_output = tk.Text(root, height=10, width=50)
transcript_output.grid(row=1, column=3, columnspan=2, padx=10, pady=10, sticky='ew')  # Spanned across 2 columns for alignment
transcript_output.config(state=tk.DISABLED)  # Disable editing

# Scrollbar for the transcript output
transcript_scrollbar = tk.Scrollbar(root, command=transcript_output.yview)
transcript_scrollbar.grid(row=1, column=5, sticky='ns')  # Adjusted sticky to 'ns' for vertical alignment
transcript_output['yscrollcommand'] = transcript_scrollbar.set

# Create the avatar label
avatar_label = tk.Label(root)
avatar_label.grid(row=1, column=6, padx=10, pady=10)

# Update the avatar with the initial state
update_avatar("thinking")


check_queue()

root.mainloop()
