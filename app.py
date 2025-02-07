import gradio as gr
from ollama import chat

LLM_MODEL = "llama3.2:1b" # RENAME WITH YOUR OWN MODEL HERE AFTER DOWNLOADING FROM OLLAMA.

def respond(prompt: str, history):
    if not history:
        history = [{"role": "system", "content": "You are a friendly chatbot."}]
    history.append({"role": "user", "content": prompt})
    response = {"role": "assistant", "content": ""}

    stream = chat(
        model=LLM_MODEL,
        messages=history,
        stream=True,
    )

    for chunk in stream:
        response["content"] += chunk['message']['content'] or ""
        yield history + [response]
    if response["content"]:
        yield history + [response]

with gr.Blocks() as demo:
    gr.Markdown("Own LLM 🦙")
    
    chatbot = gr.Chatbot(
        label="Agent",
        type="messages",
        avatar_images=(
            None,
            "assets/logo/logo.png",
        ),
    )

    prompt = gr.Textbox(max_lines=1, label="Chat Message", placeholder="Type your message here")
    prompt.submit(respond, [prompt, chatbot], [chatbot])
    prompt.submit(lambda: "", None, [prompt])

if __name__ == "__main__":
    demo.launch()
