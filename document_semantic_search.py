import yaml
import argparse
import gradio as gr
from rag import ContextManager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semantic_search.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    cm = ContextManager(config)

    with gr.Blocks() as demo:
        file_output = gr.Files()
        upload_button = gr.UploadButton(
            "Click to Upload Files",
            file_count="multiple",
        )
        sources_box = gr.TextArea(cm.get_sources, label="Sources")
        
        context_box = gr.TextArea(label="Context")
        msg = gr.Textbox(label="Input")
        clear = gr.Button("Clear")

        upload_button.upload(cm.upload_file, upload_button, file_output).then(
            cm.get_sources, None, sources_box
        )
        msg.submit(cm.get_context, msg, context_box, queue=False)
        clear.click(lambda: None, None, context_box, queue=False)

    demo.queue().launch()
