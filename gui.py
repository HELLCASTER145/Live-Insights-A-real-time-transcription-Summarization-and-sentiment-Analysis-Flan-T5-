import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.simpledialog import askfloat
import threading
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import sounddevice as sd
import wavio
from datetime import datetime
import whisper
import shutil
from pydub import AudioSegment

# Initialize models and pipelines
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summarization_model_name = 'C:/Users/laljm/flan-t5-base'
summarization_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(summarization_model_name, use_fast=True)
summarizer = pipeline(task="summarization", model=summarization_model, tokenizer=tokenizer, device=0 if device.type == "cuda" else -1)

sentiment_model = AutoModelForSeq2SeqLM.from_pretrained(summarization_model_name, torch_dtype=torch.bfloat16).to(device)
sentiment_tokenizer = AutoTokenizer.from_pretrained(summarization_model_name, use_fast=True)
sentiment_analyzer = pipeline(task="text2text-generation", model=sentiment_model, tokenizer=sentiment_tokenizer, device=0 if device.type == "cuda" else -1)

def record_audio(duration):
    freq = 44100
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    sd.wait()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_wavio = f"recording_wavio_{timestamp}.wav"
    wavio.write(filename_wavio, recording, freq, sampwidth=2)
    print(f"Recording saved as {filename_wavio}.")
    return filename_wavio

def convert_mp3_to_wav(mp3_path):
    wav_path = mp3_path.replace('.mp3', '.wav')
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

def transcribe_existing_audio(file_path):
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("ffmpeg is not installed")
    
    if file_path.lower().endswith('.mp3'):
        file_path = convert_mp3_to_wav(file_path)
    
    model = whisper.load_model("medium")
    result = model.transcribe(file_path)
    transcription_text = result["text"]
    return transcription_text

def summarize_text(text, prompt, batch_size=512):
    formatted_prompt = f"{prompt}\n\n{text}"
    
    input_ids = tokenizer.encode(formatted_prompt, truncation=True)
    
    num_batches = len(input_ids) // batch_size
    if len(input_ids) % batch_size != 0:
        num_batches += 1
    
    summaries = []
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = (i + 1) * batch_size
        input_ids_batch = input_ids[start_idx:end_idx]
        
        text_batch = tokenizer.decode(input_ids_batch, skip_special_tokens=True)
        
        try:
            summary = summarizer(text_batch, max_length=150, min_length=80, do_sample=False)
            if summary and isinstance(summary, list) and len(summary) > 0:
                summaries.append(summary[0]['summary_text'])
            else:
                summaries.append("No summary available.")
        except Exception as e:
            print(f"Error during summarization: {e}")
            summaries.append("Error during summarization.")
    
    final_summary = "\n".join(summaries)
    
    return final_summary

def analyze_sentiment(text):
    prompt = f"Classify the sentiment of the following text: {text}\nSentiment:"
    result = sentiment_analyzer(prompt)
    sentiment = result[0]['generated_text'].strip()
    return sentiment

def process_audio_and_text(audio_file=None, duration=None):
    try:
        if audio_file:
            transcription_text = transcribe_existing_audio(audio_file)
        elif duration:
            audio_file = record_audio(duration)
            transcription_text = transcribe_existing_audio(audio_file)
        else:
            messagebox.showerror("Error", "No audio file or duration provided.")
            return
        
        prompt = prompt_entry.get()

        summary_text = summarize_text(transcription_text, prompt)
        sentiment = analyze_sentiment(transcription_text)

        # Update GUI in the main thread
        def update_gui():
            transcription_text_box.config(state=tk.NORMAL)
            transcription_text_box.delete(1.0, tk.END)
            transcription_text_box.insert(tk.END, transcription_text)
            transcription_text_box.config(state=tk.DISABLED)
            
            summary_text_box.config(state=tk.NORMAL)
            summary_text_box.delete(1.0, tk.END)
            summary_text_box.insert(tk.END, summary_text)
            summary_text_box.config(state=tk.DISABLED)

            sentiment_label.config(text=f"Sentiment: {sentiment}")
            
            messagebox.showinfo("Processing Complete", "Audio processed successfully!")

        # Use the Tkinter main thread to update GUI
        root.after(0, update_gui)

    except Exception as e:
        # Use the Tkinter main thread to show error
        root.after(0, lambda: messagebox.showerror("Error", str(e)))

def start_processing():
    if input_audio_var.get() == 1:
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav;*.mp3")])
        if file_path:
            threading.Thread(target=process_audio_and_text, args=(file_path,)).start()
    else:
        duration = askfloat("Input Duration", "Enter duration for recording (in seconds):", minvalue=1.0)
        if duration:
            threading.Thread(target=process_audio_and_text, args=(None, duration)).start()

# Create the GUI
root = tk.Tk()
root.title("Audio Processing and Summarization")
root.configure(bg="#3c3c3b")  # Dark olive green background

# GUI Elements
input_audio_var = tk.IntVar()
input_audio_radio = tk.Radiobutton(root, text="Input Existing Audio File", variable=input_audio_var, value=1, bg="#3c3c3b", fg="#ffffff")
input_audio_radio.pack(pady=5)

record_audio_radio = tk.Radiobutton(root, text="Record New Audio", variable=input_audio_var, value=0, bg="#3c3c3b", fg="#ffffff")
record_audio_radio.pack(pady=5)

process_button = tk.Button(root, text="Process Audio", command=start_processing, bg="#4a4a48", fg="#ffffff")
process_button.pack(pady=10)

prompt_label = tk.Label(root, text="Enter a prompt for summarization:", bg="#3c3c3b", fg="#ffffff")
prompt_label.pack(pady=5)
prompt_entry = tk.Entry(root, width=50)
prompt_entry.pack(pady=5)

transcription_label = tk.Label(root, text="Transcription:", bg="#3c3c3b", fg="#ffffff")
transcription_label.pack(pady=5)
transcription_text_box = tk.Text(root, height=10, width=60, wrap=tk.WORD, state=tk.DISABLED)
transcription_text_box.pack(pady=5)

summary_label = tk.Label(root, text="Summary:", bg="#3c3c3b", fg="#ffffff")
summary_label.pack(pady=5)
summary_text_box = tk.Text(root, height=10, width=60, wrap=tk.WORD, state=tk.DISABLED)
summary_text_box.pack(pady=5)

sentiment_label = tk.Label(root, text="Sentiment:", bg="#3c3c3b", fg="#ffffff")
sentiment_label.pack(pady=5)

# Start the GUI main loop
root.mainloop()
