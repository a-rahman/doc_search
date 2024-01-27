import gradio as gr
from rag import ContextManager

if __name__ == "__main__":
    cm = ContextManager()

    with gr.Blocks() as demo:
        context_box = gr.TextArea(label="Context")
        msg = gr.Textbox(label="Input")
        clear = gr.Button("Clear")
        sources_box = gr.TextArea(cm.get_sources, label="Sources")

        file_output = gr.Files()
        upload_button = gr.UploadButton(
            "Click to Upload Files",
            file_count="multiple",
        )

        upload_button.upload(cm.upload_file, upload_button, file_output).then(
            cm.get_sources, None, sources_box
        )
        msg.submit(cm.get_context, msg, context_box, queue=False)
        clear.click(lambda: None, None, context_box, queue=False)

    demo.queue().launch(server_name="0.0.0.0")
