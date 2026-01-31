"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import time

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
use_kvcache = False # whether or not to use kvcache
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

if not use_kvcache:
    from model import GPTConfig, GPT
else:
    from model_kv import GPTConfig, GPT

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# run generation
with torch.no_grad():
    with ctx:
        start = time.time_ns()
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            # POC - batch inference works
            print("Gen 1\n*****\n")
            print(decode(y[0].tolist()))
            print("Gen length -", len(y[0].tolist()))
            print("Gen 2\n*****\n")
            print(decode(y[1].tolist()))
            print("Gen length -", len(y[1].tolist()))
            print('---------------')
        end = time.time_ns()

print(f'{(end-start)/1_000_000}ms')

# run generation
# with torch.no_grad():
#     with ctx:
#         for k in range(num_samples):
#             y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
#             print(decode(y[0].tolist()))
#             print('---------------')


# ======== Measure inference time vs input length ========

# base_prompt = "The quick brown fox jumps over the lazy dog. "

# results = []

# torch.cuda.reset_peak_memory_stats()

# with torch.no_grad():
#     with ctx:
#         for k in [1, 5, 10, 20, 40, 80]:     # k controls input length
#             start = base_prompt * k
#             x = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]
            
#             torch.cuda.synchronize()
#             start_time = time.time()

#             y = model.generate(
#                 x,
#                 max_new_tokens,
#                 temperature=temperature,
#                 top_k=top_k
#             )

#             torch.cuda.synchronize()
#             end_time = time.time()
            
#             total_time = end_time - start_time
#             num_generated_tokens = y.shape[1] - x.shape[1]
            
#             per_token_time = total_time / num_generated_tokens

#             results.append((len(x[0]), total_time, per_token_time))
    
#             print(decode(y[0].tolist()))
#             print('---------------')

# peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
# print(f"Peak GPU memory usage: {peak_mem:.2f} GB")
# print("\nInput Tokens | Total Time (s) | Time per Token (s)")
# for input_len, total_t, per_token_t in results:
#     print(f"{input_len:12} | {total_t:.4f}       | {per_token_t:.6f}")
    
# # plot the inference time vs input length
# import matplotlib.pyplot as plt
# input_lengths = [r[0] for r in results]
# total_times = [r[1] for r in results]
# per_token_times = [r[2] for r in results]
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(input_lengths, total_times, marker='o')
# plt.title('Total Inference Time vs Input Length')
# plt.xlabel('Input Length (tokens)')
# plt.ylabel('Total Inference Time (s)')
# plt.subplot(1, 2, 2)
# plt.plot(input_lengths, per_token_times, marker='o', color='orange')
# plt.title('Per Token Inference Time vs Input Length')
# plt.xlabel('Input Length (tokens)')
# plt.ylabel('Per Token Inference Time (s)')
# plt.tight_layout()
# plt.savefig('inference_time_vs_input_length.png')

# -----------------------------------------------------------------------------


# # ======== Measure inference time vs output length ========
# fixed_prompt = "Explain transformers in simple terms."
# x = torch.tensor(encode(fixed_prompt), dtype=torch.long, device=device)[None, ...]

# results = []

# torch.cuda.reset_peak_memory_stats()

# with torch.no_grad():
#     with ctx:
#         for out_len in [50, 100, 200, 400, 800]:     # out_len controls output length
#             torch.cuda.synchronize()
#             start_time = time.time()

#             y = model.generate(
#                 x,
#                 max_new_tokens=out_len,
#                 temperature=temperature,
#                 top_k=top_k
#             )

#             torch.cuda.synchronize()
#             end_time = time.time()
            
#             total_time = end_time - start_time
#             num_generated_tokens = y.shape[1] - x.shape[1]
            
#             per_token_time = total_time / num_generated_tokens

#             results.append((out_len, total_time, per_token_time))
    
#             print(decode(y[0].tolist()))
#             print('---------------')
    
# peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
# print(f"Peak GPU memory usage: {peak_mem:.2f} GB")
# print("\nOutput Tokens | Total Time (s) | Time per Token (s)")
# for out_len, total_t, per_token_t in results:
#     print(f"{out_len:12} | {total_t:.4f}       | {per_token_t:.6f}")
    
# # plot the inference time vs output length
# import matplotlib.pyplot as plt
# output_lengths = [r[0] for r in results]
# total_times = [r[1] for r in results]
# per_token_times = [r[2] for r in results]
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(output_lengths, total_times, marker='o')
# plt.title('Total Inference Time vs Output Length')
# plt.xlabel('Output Length (tokens)')
# plt.ylabel('Total Inference Time (s)')
# plt.subplot(1, 2, 2)
# plt.plot(output_lengths, per_token_times, marker='o', color='orange')
# plt.title('Per Token Inference Time vs Output Length')
# plt.xlabel('Output Length (tokens)')
# plt.ylabel('Per Token Inference Time (s)')
# plt.tight_layout()
# plt.savefig('inference_time_vs_output_length.png')
# # -----------------------------------------------------------------------------