import gradio as gr
from ollama import chat
import json
import os
import threading

LLM_MODEL = "llama3.2:1b"  # Replace with your model name
HISTORY_FILE = "chat_history.json"
stop_event = threading.Event()

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as file:
            return json.load(file)
    return [{"role": "system", "content": "You are a friendly chatbot."}]

def save_history(history):
    with open(HISTORY_FILE, "w") as file:
        json.dump(history, file)

def respond(prompt: str, history):
    if not history:
        history = load_history()
    history.append({"role": "user", "content": prompt})
    response = {"role": "assistant", "content": ""}

    stream = chat(
        model=LLM_MODEL,
        messages=history,
        stream=True,
    )

    for chunk in stream:
        if stop_event.is_set():
            stop_event.clear()
            break
        response["content"] += chunk['message']['content'] or ""
        yield history + [response]
    if response["content"]:
        save_history(history + [response])
        yield history + [response]

def stop_response():
    stop_event.set()

with gr.Blocks() as demo:
    gr.Markdown("Own LLM 🦙")
    
    chatbot = gr.Chatbot(
        label="Howdy",
        type="messages",
        avatar_images=(
            None,
            "logo.png",
        ),
        value=load_history()

    )

    with gr.Row():
        prompt = gr.Textbox(placeholder="How can i help you...", scale=10)
        stop_btn = gr.Button("Stop", elem_id="stop-button", scale=1)


    prompt.submit(respond, [prompt, chatbot], [chatbot])
    prompt.submit(lambda: "", None, [prompt])
    stop_btn.click(stop_response)

    demo.css = """
    #stop-button {
        width: 50px;
    }
    """

if __name__ == "__main__":
    demo.queue().launch()
