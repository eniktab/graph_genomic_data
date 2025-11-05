from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os
import random

DNA = "ACGT"



# Hard-disable any hub/network lookups
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_DIR = "/g/data/te53/en9803/data/hf-cache/models/nucleotide-transformer-2.5b-1000g"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
torch.set_default_device("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    dtype=dtype,
    device_map="auto",     # places layers across available GPUs/CPU
    low_cpu_mem_usage=True # reduces CPU RAM during load
)

# Put the whole model on one device to avoid sharding surprises
device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()


# Choose the length to which the input sequences are padded. By default, the
# model max length is chosen, but feel free to decrease it as the time taken to
# obtain the embeddings increases significantly with it.
max_length = tokenizer.model_max_length

rng = random.Random(1337)
def rand_dna(n: int, rng: random.Random) -> str:
    return "".join(rng.choice(DNA) for _ in range(n))
# Create a dummy dna sequence and tokenize it
a = rand_dna(600, rng)
sequences = [ a[:1], a , a+a, 10*a[:5994]]

tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", truncation=True,
                                         pad_to_multiple_of=8,
                                         padding="max_length", max_length = max_length)["input_ids"]

tokens_ids[-1]

# Compute the embeddings
attention_mask = tokens_ids != tokenizer.pad_token_id
torch_outs = model(
    tokens_ids,
    attention_mask=attention_mask,
    encoder_attention_mask=attention_mask,
    output_hidden_states=True
)


# Compute sequences embeddings
embeddings = torch_outs['hidden_states'][-1].detach()
print(f"Embeddings shape: {embeddings.shape}")
print(f"Embeddings per token: {embeddings}")

# Add embed dimension axis
attention_mask = torch.unsqueeze(attention_mask, dim=-1)

# Compute mean embeddings per sequence
mean_sequence_embeddings = torch.sum(attention_mask*embeddings, axis=-2)/torch.sum(attention_mask, axis=1)
print(f"Mean sequence embeddings: {mean_sequence_embeddings}")
print(mean_sequence_embeddings.shape)

