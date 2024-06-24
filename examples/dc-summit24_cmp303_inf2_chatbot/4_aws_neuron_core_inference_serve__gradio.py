import os
import boto3
import torch
from ray import serve
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split
from ray.serve.gradio_integrations import GradioIngress
import gradio as gr

# Create Secrets Manager clients
secrets_manager = boto3.client('secretsmanager', region_name='us-east-1')

# Retrieve the secret from Secrets Manager
secret_name = 'HFaccessToken'
response = secrets_manager.get_secret_value(SecretId=secret_name)
access_token = response['SecretString']

hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"
local_model_path = f"{hf_model.replace('/','_')}-split"


@serve.deployment(
    ray_actor_options={
        "resources": {"neuron_cores": 12},
        "runtime_env": {
            "env_vars": {
                "NEURON_CC_FLAGS": "-O1",
                "NEURON_COMPILE_CACHE_URL": "/home/ubuntu/neuron_demo/neuron-compile-cache",
            }
        },
    },
    num_replicas=1,
)
class LlamaModel:
    def __init__(self):
        if not os.path.exists(local_model_path):
            print(f"Saving model split for {hf_model} to local path {local_model_path}")
            self.model = LlamaForCausalLM.from_pretrained(hf_model, token=access_token)
            save_pretrained_split(self.model, local_model_path)
        else:
            print(f"Using existing model split {local_model_path}")

        print(f"Loading and compiling model {local_model_path} for Neuron")
        self.neuron_model = LlamaForSampling.from_pretrained(
            local_model_path, batch_size=1, tp_degree=12, amp="f16"
        )
        self.neuron_model.to_neuron()
        self.tokenizer = AutoTokenizer.from_pretrained(hf_model, token=access_token)

    def infer(self, sentence: str):
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
        with torch.inference_mode():
            generated_sequences = self.neuron_model.sample(
                input_ids, sequence_length=512, top_k=20
            )
        return "\n".join([self.tokenizer.decode(seq) for seq in generated_sequences])


@serve.deployment
class MyGradioServer(GradioIngress):
    def __init__(self, downstream_handle):
        self.downstream = downstream_handle.options(use_new_handle_api=True)

        with gr.Blocks() as demo:
            gr.Markdown("## Simple LLM Chatbot")
            gr.Markdown("Enter a prompt for Meta Llama 3")
            with gr.Row():
                inp = gr.Textbox(label="Input prompt:")
            with gr.Row():
                btn = gr.Button("Generate")
            with gr.Row():
                out = gr.Textbox(label="Llama 3 output:", lines=30)
            btn.click(
                fn=self.do_update,
                inputs=inp,
                outputs=out,
            )

        super().__init__(lambda: demo)

    async def do_update(self, txt):
        return await self.downstream.infer.remote(txt)


llama_model = LlamaModel.bind()
app = MyGradioServer.bind(llama_model)
