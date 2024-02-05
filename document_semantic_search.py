import os
import glob
import yaml
import argparse
import gradio as gr
from rag import ContextManager


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic_search.yaml")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    cm = ContextManager(config)

    def dropdown_list():
        dirs = glob.glob("doc_index/*")
        return gr.Dropdown(choices=[os.path.basename(dir) for dir in dirs], interactive=True)

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                file_output = gr.Files()
                upload_button = gr.UploadButton(
                    "Click to Upload Files",
                    file_count="multiple",
                )
            with gr.Column(scale=1, min_width=600):
                dirs = glob.glob("doc_index/*")
                db_choice = gr.Dropdown(
                    [os.path.basename(dir) for dir in dirs], label="Databases", allow_custom_value=True
                )
                db_refresh = gr.Button("Refresh Database")

        sources_box = gr.TextArea(cm.get_sources, label="Sources")

        context_box = gr.TextArea(label="Context")
        msg = gr.Textbox(label="Input")
        clear = gr.Button("Clear")

        upload_button.upload(
            cm.upload_file, [upload_button, db_choice], file_output
        ).then(cm.get_sources, None, sources_box)
        msg.submit(cm.get_context, msg, context_box, queue=False)
        clear.click(lambda: None, None, context_box, queue=False)
        db_refresh.click(cm.change_db, db_choice, None).then(
            cm.get_sources, None, sources_box
        ).then(dropdown_list, None, db_choice)

    demo.queue().launch()
