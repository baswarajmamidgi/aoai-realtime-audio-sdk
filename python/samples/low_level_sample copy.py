
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import base64
import os
import sys

import numpy as np
import soundfile as sf
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
from scipy.signal import resample
import pyaudio
import wave
import tkinter as tk
import threading
import sounddevice as sd

from rtclient import (
    InputAudioBufferAppendMessage,
    InputAudioTranscription,
    RTLowLevelClient,
    ServerVAD,
    SessionUpdateMessage,
    SessionUpdateParams,
)




async def receive_messages(client: RTLowLevelClient):
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
                break
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
            case "response.audio.done":
                print("Response Audio Done Message")
                print(f"  Response Id: {message.response_id}")
                print(f"  Item Id: {message.item_id}")
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


async def with_azure_openai(audio_file_path: str):
    endpoint = get_env_var("AZURE_OPENAI_ENDPOINT")
    key = get_env_var("AZURE_OPENAI_API_KEY")
    deployment = get_env_var("AZURE_OPENAI_DEPLOYMENT")
    async with RTLowLevelClient(
        endpoint, key_credential=AzureKeyCredential(key), azure_deployment=deployment
    ) as client:
        await client.send(
            SessionUpdateMessage(
                session=SessionUpdateParams(
                    turn_detection=ServerVAD(type="server_vad"),
                    input_audio_transcription=InputAudioTranscription(model="whisper-1"),
                )
            )
        )

        await asyncio.gather(send_audio(client, audio_file_path), receive_messages(client))




class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.sample_rate = 16000
        self.chunk = 4800
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.sample_rate,
                                      input=True,
                                      frames_per_buffer=self.chunk)
        print("Recording started...")

    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.is_recording = False
        print("Recording stopped.")

    def get_audio_data(self):
        return b''.join(self.frames)

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Recorder")
        self.recorder = AudioRecorder()
        self.button = tk.Button(self, text="Start Recording", command=self.toggle_recording)
        self.button.pack(pady=20)
        self.loop = asyncio.new_event_loop()
        self.api_thread = None
        self.status_label = tk.Label(self, text="")
        self.status_label.pack(pady=10)

    def toggle_recording(self):
        if not self.recorder.is_recording:
            self.recorder.start_recording()
            self.button.config(text="Stop Recording")
            threading.Thread(target=self.record_audio).start()
        else:
            self.recorder.stop_recording()
            self.button.config(text="Start Recording")
            self.process_audio()

    def record_audio(self):
        try:
            while self.recorder.is_recording:
                data = self.recorder.stream.read(self.recorder.chunk)
                self.recorder.frames.append(data)
        except Exception as e:
            self.show_status(f"Error during recording: {str(e)}")

    def process_audio(self):
        try:
            audio_data = self.recorder.get_audio_data()
            if self.api_thread and self.api_thread.is_alive():
                self.loop.call_soon_threadsafe(self.loop.stop)
                self.api_thread.join()
            self.api_thread = threading.Thread(target=self.run_async_tasks, args=(audio_data,))
            self.api_thread.start()
        except Exception as e:
            self.show_status(f"Error processing audio: {str(e)}")

    def run_async_tasks(self, audio_data):
        try:
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.send_audio_to_api(audio_data))
        except Exception as e:
            self.show_status(f"Error in API communication: {str(e)}")

    async def send_audio_to_api(self, audio_data):
        try:
            endpoint = get_env_var("AZURE_OPENAI_ENDPOINT")
            key = get_env_var("AZURE_OPENAI_API_KEY")
            deployment = get_env_var("AZURE_OPENAI_DEPLOYMENT")

            async with RTLowLevelClient(
                endpoint, key_credential=AzureKeyCredential(key), azure_deployment=deployment
            ) as client:
                await client.send(
                    SessionUpdateMessage(
                        session=SessionUpdateParams(
                            turn_detection=ServerVAD(type="server_vad"),
                            input_audio_transcription=InputAudioTranscription(model="whisper-1"),
                        )
                    )
                )

                await asyncio.gather(
                    self.send_audio(client, audio_data),
                    self.receive_and_play_response(client)
                )
        except Exception as e:
            self.show_status(f"Error in API communication: {str(e)}")

    async def send_audio(self, client: RTLowLevelClient, audio_data: bytes):
        try:
            chunk_size = 4800  # 100ms at 24000 Hz, 2 bytes per sample
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                base64_audio = base64.b64encode(chunk).decode("utf-8")
                await client.send(InputAudioBufferAppendMessage(audio=base64_audio))
            # Send end of stream message
            # await client.send(InputAudioBufferAppendMessage(audio=None))
        except Exception as e:
            self.show_status(f"Error sending audio: {str(e)}")

    async def receive_and_play_response(self, client: RTLowLevelClient):
        try:
            audio_buffer = []
            while not client.closed:
                message = await client.recv()
                if message is None:
                    continue
                if message.type == "response.audio.delta":
                    audio_buffer.append(base64.b64decode(message.delta))
                elif message.type == "response.done":
                    print("Response done message")
                    if audio_buffer:
                        audio_data = b''.join(audio_buffer)
                        self.after(0, self.play_audio, audio_data)
                    break
                elif message.type == "response.text.delta":
                    self.show_status(f"Received text: {message.delta}")
        except Exception as e:
            self.show_status(f"Error receiving response: {str(e)}")

    def play_audio(self, audio_data):
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            sd.play(audio_array, samplerate=24000)
            sd.wait()
        except Exception as e:
            self.show_status(f"Error playing audio: {str(e)}")

    def show_status(self, message):
        print(message)  # Print to console for debugging
        self.status_label.config(text=message)

if __name__ == "__main__":
    load_dotenv()
    try:
        app = Application()
        app.mainloop()
    except Exception as e:
        print(f"Unhandled exception: {str(e)}")
