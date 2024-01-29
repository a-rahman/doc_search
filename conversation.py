import yaml
import argparse
import torch
import gradio as gr
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
)
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

from rag import ContextManager

qna_prompt_template = """### [INST] Instruction: You will be provided with questions and related data. Your task is to find the answers to the questions using the given data. If the data doesn't contain the answer to the question, then you must return 'Not enough information.'

{context}

### Question: {question} [/INST]"""

PROMPT = PromptTemplate(
    template=qna_prompt_template, input_variables=["context", "question"]
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/semanic_search.yaml")
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    cm = ContextManager(config)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(config["llm"]["model"])
    model = AutoModelForCausalLM.from_pretrained(
        config["llm"]["model"],
        device_map="auto",
        quantization_config=quantization_config,
    )

    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=300
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

    def ask(question):
        context = cm.retriever.get_relevant_documents(question)
        answer = (
            chain(
                {"input_documents": context, "question": question},
                return_only_outputs=True,
            )
        )["output_text"]
        return answer, cm.format_context(context)

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Input")
        context_button = gr.Button("Get Context Only")
        clear = gr.Button("Clear")
        context_box = gr.TextArea(label="Context")

        def user(user_message, history):
            return "", history + [[user_message, None]]

        def bot(history):
            bot_message, context = ask(history[-1][0])
            history[-1][1] = bot_message
            return history, context

        file_output = gr.File()
        upload_button = gr.UploadButton(
            "Click to Upload Files",
            file_count="multiple",
        )
        upload_button.upload(cm.upload_file, upload_button, file_output)

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, [chatbot, context_box]
        )
        context_button.click(cm.get_context, msg, context_box)
        clear.click(lambda: [None, None], None, [chatbot, context_box], queue=False)

    demo.queue().launch(server_name="0.0.0.0")
