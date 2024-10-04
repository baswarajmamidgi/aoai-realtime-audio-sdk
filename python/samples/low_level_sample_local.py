# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import os
import sys
import tkinter as tk
import pyaudio
import threading
import queue
import io
import wave

import numpy as np
import soundfile as sf
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from scipy.signal import resample

from rtclient import (
    InputAudioBufferAppendMessage,
    InputAudioTranscription,
    RTLowLevelClient,
    ServerVAD,
    SessionUpdateMessage,
    SessionUpdateParams,
)




def play_audio(audio_data):
    CHUNK = 1024
    
    # Create a wave file in memory
    with io.BytesIO() as wav_buffer:
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)  # Assuming 16-bit audio
            wav_file.setframerate(24000)  # Assuming 24kHz sample rate
            wav_file.writeframes(audio_data)
        
        wav_buffer.seek(0)
        
        # Play the audio
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(2),
                        channels=1,
                        rate=24000,
                        output=True)
        
        wav_buffer.seek(44)  # Skip WAV header
        data = wav_buffer.read(CHUNK)
        
        while data:
            stream.write(data)
            data = wav_buffer.read(CHUNK)
        
        stream.stop_stream()
        stream.close()
        p.terminate()


async def receive_messages(client: RTLowLevelClient):
    audio_buffer = b''
    while not client.closed:
        message = await client.recv()
        if message is None:
            continue
        match message.type:
            case "session.created":
                print("Session Created Message")
                print(f"  Model: {message.session.model}")
                print(f"  Session Id: {message.session.id}")
                pass
            case "error":
                print("Error Message")
                print(f"  Error: {message.error}")
                pass
            case "input_audio_buffer.committed":
                print("Input Audio Buffer Committed Message")
                print(f"  Item Id: {message.item_id}")
                pass
            case "input_audio_buffer.cleared":
                print("Input Audio Buffer Cleared Message")
                pass
            case "input_audio_buffer.speech_started":
                print("Input Audio Buffer Speech Started Message")
                print(f"  Item Id: {message.item_id}")
                print(f"  Audio Start [ms]: {message.audio_start_ms}")
                pass
            case "input_audio_buffer.speech_stopped":
                print("Input Audio Buffer Speech Stopped Message")
                print(f"  Item Id: {message.item_id}")
                print(f"  Audio End [ms]: {message.audio_end_ms}")
                pass
            case "conversation.item.created":
                print("Conversation Item Created Message")
                print(f"  Id: {message.item.id}")
                print(f"  Previous Id: {message.previous_item_id}")
                if message.item.type == "message":
                    print(f"  Role: {message.item.role}")
                    for index, content in enumerate(message.item.content):
                        print(f"  [{index}]:")
                        print(f"    Content Type: {content.type}")
                        if content.type == "input_text" or content.type == "text":
                            print(f"  Text: {content.text}")
                        elif content.type == "input_audio" or content.type == "audio":
                            print(f"  Audio Transcript: {content.transcript}")
                pass
            case "conversation.item.truncated":
                print("Conversation Item Truncated Message")
                print(f"  Id: {message.item_id}")
                print(f" Content Index: {message.content_index}")
                print(f"  Audio End [ms]: {message.audio_end_ms}")
            case "conversation.item.deleted":
                print("Conversation Item Deleted Message")
                print(f"  Id: {message.item_id}")
            case "conversation.item.input_audio_transcription.completed":
                print("Input Audio Transcription Completed Message")
                print(f"  Id: {message.item_id}")
                print(f"  Content Index: {message.content_index}")
                print(f"  Transcript: {message.transcript}")
            case "conversation.item.input_audio_transcription.failed":
                print("Input Audio Transcription Failed Message")
                print(f"  Id: {message.item_id}")
                print(f"  Error: {message.error}")
            case "response.created":
                print("Response Created Message")
                print(f"  Response Id: {message.response.id}")
                print("  Output Items:")
                for index, item in enumerate(message.response.output):
                    print(f"  [{index}]:")
                    print(f"    Item Id: {item.id}")
                    print(f"    Type: {item.type}")
                    if item.type == "message":
                        print(f"    Role: {item.role}")
                        match item.role:
                            case "system":
                                for content_index, content in enumerate(item.content):
                                    print(f"    [{content_index}]:")
                                    print(f"      Content Type: {content.type}")
                                    print(f"      Text: {content.text}")
                            case "user":
                                for content_index, content in enumerate(item.content):
                                    print(f"    [{content_index}]:")
                                    print(f"      Content Type: {content.type}")
                                    if content.type == "input_text":
                                        print(f"      Text: {content.text}")
                                    elif content.type == "input_audio":
                                        print(f"      Audio Data Length: {len(content.audio)}")
                            case "assistant":
                                for content_index, content in enumerate(item.content):
                                    print(f"    [{content_index}]:")
                                    print(f"      Content Type: {content.type}")
                                    print(f"      Text: {content.text}")
                    elif item.type == "function_call":
                        print(f"    Call Id: {item.call_id}")
                        print(f"    Function Name: {item.name}")
                        print(f"    Parameters: {item.arguments}")
                    elif item.type == "function_call_output":
                        print(f"    Call Id: {item.call_id}")
                        print(f"    Output: {item.output}")
            case "response.done":
                print("Response Done Message")
                print(f"  Response Id: {message.response.id}")
                if message.response.status_details:
                    print(f"  Status Details: {message.response.status_details.model_dump_json()}")
                # break
            case "response.output_item.added":
                print("Response Output Item Added Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item.id}")
            case "response.output_item.done":
                print("Response Output Item Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item.id}")

            case "response.content_part.added":
                print("Response Content Part Added Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item_id}")
            case "response.content_part.done":
                print("Response Content Part Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  ItemPart Id: {message.item_id}")
            case "response.text.delta":
                print("Response Text Delta Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Text: {message.delta}")
            case "response.text.done":
                print("Response Text Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Text: {message.text}")
            case "response.audio_transcript.delta":
                print("Response Audio Transcript Delta Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item_id}")
                print(f"  Transcript: {message.delta}")
            case "response.audio_transcript.done":
                print("Response Audio Transcript Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item_id}")
                print(f"  Transcript: {message.transcript}")
            case "response.audio.delta":
                print("Response Audio Delta Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item_id}")
                print(f"  Audio Data Length: {len(message.delta)}")
                audio_buffer += base64.b64decode(message.delta)

            case "response.audio.done":
                print("Response Audio Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item_id}")
                # Play the complete audio buffer
                play_audio(audio_buffer)
                audio_buffer = b''  # Reset the buffer
                
            case "response.function_call_arguments.delta":
                print("Response Function Call Arguments Delta Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Arguments: {message.delta}")
            case "response.function_call_arguments.done":
                print("Response Function Call Arguments Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Arguments: {message.arguments}")
            case "rate_limits.updated":
                print("Rate Limits Updated Message")
                print(f"  Rate Limits: {message.rate_limits}")
            case _:
                print("Unknown Message")


def get_env_var(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise OSError(f"Environment variable '{var_name}' is not set or is empty.")
    return value


async def with_azure_openai(audio_queue: queue.Queue, stop_event: threading.Event):
    endpoint = get_env_var("AZURE_OPENAI_ENDPOINT")
    key = get_env_var("AZURE_OPENAI_API_KEY")
    deployment = get_env_var("AZURE_OPENAI_DEPLOYMENT")
    async with RTLowLevelClient(
        endpoint, key_credential=AzureKeyCredential(key), azure_deployment=deployment
    ) as client:
        # Send the session update message once
        await client.send(
            SessionUpdateMessage(
                session=SessionUpdateParams(
                    turn_detection=ServerVAD(type="server_vad"),
                    input_audio_transcription=InputAudioTranscription(model="whisper-1"),
                )
            )
        )

        # Start streaming audio and receiving messages concurrently
        await asyncio.gather(stream_audio(client, audio_queue, stop_event), receive_messages(client))


async def stream_audio(client: RTLowLevelClient, audio_queue: queue.Queue, stop_event: threading.Event):
    while not stop_event.is_set():
        try:
            chunk = audio_queue.get_nowait()
            base64_audio = base64.b64encode(chunk).decode("utf-8")
            await client.send(InputAudioBufferAppendMessage(audio=base64_audio))
        except queue.Empty:
            await asyncio.sleep(0.1)  # Short sleep to prevent busy-waiting


def record_audio(audio_queue: queue.Queue, stop_event: threading.Event):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 24000

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")

    while not stop_event.is_set():
        data = stream.read(CHUNK)
        print("Recoding In Progress")
        audio_queue.put(data)

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()


def create_gui():
    root = tk.Tk()
    root.title("Audio Recorder")

    recording = False
    stop_event = threading.Event()
    audio_queue = queue.Queue()
    processing_thread = None

    def toggle_recording():
        nonlocal recording, processing_thread
        if not recording:
            recording = True
            button.config(text="Stop Recording")
            stop_event.clear()
            # Start the recording thread
            threading.Thread(target=record_audio, args=(audio_queue, stop_event), daemon=True).start()
            # Start the processing thread
            processing_thread = threading.Thread(target=run_azure_openai, args=(audio_queue, stop_event), daemon=True)
            processing_thread.start()
        else:
            recording = False
            button.config(text="Start Recording")
            stop_event.set()
            if processing_thread:
                processing_thread.join()

    button = tk.Button(root, text="Start Recording", command=toggle_recording)
    button.pack(pady=20)

    def on_closing():
        if recording:
            stop_event.set()
        root.quit()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    return root, audio_queue


def run_azure_openai(audio_queue, stop_event):
    asyncio.run(with_azure_openai(audio_queue, stop_event))


if __name__ == "__main__":
    load_dotenv()
    
    root, audio_queue = create_gui()
    root.mainloop()