import os
import torch
import boto3
from ray import serve
from starlette.requests import Request
from transformers import AutoTokenizer, AutoModelForCausalLM #LlamaForCausalLM
from transformers_neuronx.llama.model import LlamaForSampling
from transformers_neuronx.module import save_pretrained_split

# Create Secrets Manager clients
secrets_manager = boto3.client('secretsmanager', region_name='us-east-1')

# Retrieve the secret from Secrets Manager
secret_name = 'HFaccessToken'
response = secrets_manager.get_secret_value(SecretId=secret_name)
access_token = response['SecretString']

#Parameters to declare the Llama 3 model
hf_model = "meta-llama/Meta-Llama-3-8B-Instruct"
local_model_path = f"/home/ubuntu/{hf_model.replace('/','_')}-split"
tokenizer = AutoTokenizer.from_pretrained(hf_model, token=access_token)

@serve.deployment
class APIIngress:
    def __init__(self, llama_model_handle) -> None:
        self.handle = llama_model_handle

    async def __call__(self, req: Request):
        sentence = req.query_params.get("sentence")
        ref = await self.handle.infer.remote(sentence)
        return ref

@serve.deployment(
    ray_actor_options={
        "resources": {"neuron_cores": 12},
        "runtime_env": {
            "env_vars": {
                "NEURON_CC_FLAGS": "-O1",
                "NEURON_COMPILE_CACHE_URL": "/home/ubuntu/neuron-compile-cache",
            }
        },
    },
)

class LlamaModel:
    def __init__(self):
        if not os.path.exists(local_model_path):
            print(f"Saving model split for {hf_model} to local path {local_model_path}")
            self.model = AutoModelForCausalLM.from_pretrained(hf_model, token=access_token)
            save_pretrained_split(self.model, local_model_path)
        else:
            print(f"Using existing model split {local_model_path}")

        print(f"Loading and compiling model {local_model_path} for Neuron")
        self.neuron_model = LlamaForSampling.from_pretrained(
            local_model_path, batch_size=1, tp_degree=12, amp="f16"
        )
        print(f"compiling...")
        self.neuron_model.to_neuron()
        print(f"compiled!")

    def infer(self, sentence: str):
        input_ids = tokenizer.encode(sentence, return_tensors="pt")
        with torch.inference_mode():
            generated_sequences = self.neuron_model.sample(
                input_ids, sequence_length=512, top_k=20
            )
        return [tokenizer.decode(seq) for seq in generated_sequences]

app = APIIngress.bind(LlamaModel.bind())
