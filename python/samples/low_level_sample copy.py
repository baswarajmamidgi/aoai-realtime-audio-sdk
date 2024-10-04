
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




def get_env_var(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise OSError(f"Environment variable '{var_name}' is not set or is empty.")
    return value



class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.sample_rate = 24000
        self.chunk = 1024
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
