"""
Sample from the trained model with PyTorch
"""
import argparse
import glob
import math
import os
import numpy as np
import pickle
from contextlib import nullcontext
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
import json
import random
from tqdm import tqdm
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import time
from contextlib import nullcontext
from datetime import datetime
from functools import partial
from export import model_export

from tinystories import get_tokenizer_model_path

# -----------------------------------------------------------------------------
checkpoint = 'stories42M.pt'
start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 1 # number of samples to draw
max_new_tokens = 100 # number of tokens generated in each sample
temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
tokenizer = "" # override the tokenizer model path
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
#dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
dtype = "float32"
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)


def load_tokenizer():
    enc = Tokenizer()
    return enc

def process_data(movies_directory="aclImdb",
                 tokenized_train_filename="imdb_tokenized_train.bin",
                 tokenized_val_filename="imdb_tokenized_val.bin",
                 tokenized_test_filename="imdb_tokenized_test.bin",
                 splits = [0.85, 0.1, 0.05]):

    enc = load_tokenizer()

    subdirs_a = ["train", "test"]
    subdirs_b = ["pos", "neg", "unsup"]

    paths = []
    for top_level_split in subdirs_a:
        for category in subdirs_b:
            if not os.path.isdir(os.path.join(movies_directory, top_level_split, category)):
                continue
            for f in os.listdir(os.path.join(movies_directory, top_level_split, category)):
                if f.endswith(".txt") and not f.startswith("."):
                    paths.append(os.path.join(movies_directory, top_level_split, category, f))
    random.seed(0)
    random.shuffle(paths)

    train_cutoff = int(splits[0] * len(paths))
    val_cutoff = int((splits[0] + splits[1]) * len(paths))
    train_paths = paths[:train_cutoff]
    val_paths = paths[train_cutoff:val_cutoff]
    test_paths = paths[val_cutoff:]
    split_paths = [train_paths, val_paths, test_paths]
    split_tokenized_filenames = [tokenized_train_filename, tokenized_val_filename, tokenized_test_filename]
    

    for i, paths in enumerate(split_paths):
        tokenized_filename = split_tokenized_filenames[i]
        all_tokens = []
        for path in tqdm(paths):
            review = open(path).read()
            review_formatted = review.strip() + "\n"
            tokens = enc.encode(review_formatted, bos=True, eos=False)
            all_tokens.extend(tokens)
        all_tokens = np.array(all_tokens, dtype=np.uint16)
        with open(tokenized_filename, "wb") as f:
            f.write(all_tokens.tobytes())
        avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
        print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")

class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len

    def __iter__(self):
        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        # train/test split. let's use only shard 0 for test split, rest train
        shard = f"imdb_tokenized_{self.split}.bin"
        m = np.memmap(shard, dtype=np.uint16, mode="r")
        num_batches = len(m) // self.max_seq_len
        num_batches -= 1  # drop the last partial batch
        assert num_batches > 0, "this shard is way too small? investigate."
        ixs = list(range(num_batches))
        rng.shuffle(ixs)
        for ix in tqdm(ixs):
            start = ix * self.max_seq_len
            end = start + self.max_seq_len + 1
            # calling .astype will copy the data into a new numpy array, now in RAM
            chunk = torch.from_numpy((m[start:end]).astype(np.int64))
            x = chunk[:-1]
            y = chunk[1:]
            yield x, y

class Task:
    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y

def load_model(checkpoint="stories42M.pt"):
    start = "" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 1 # number of samples to draw
    max_new_tokens = 100 # number of tokens generated in each sample
    temperature = 1.0 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 300 # retain only the top_k most likely tokens, clamp others to have 0 probability
    tokenizer = "" # override the tokenizer model path
    seed = 1337
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
    #dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
    dtype = "float32"
    compile = False # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # init from a model saved in a specific directory
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)

def train_model(starting_checkpoint="stories42M.pt", out_dir="stories42M_movie_reviews"):
    # -----------------------------------------------------------------------------
    # I/O
    eval_interval = 2000
    log_interval = 10
    eval_iters = 100
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = False  # if True, always save a checkpoint after each eval
    init_from = "resume"  # 'scratch' or 'resume'
    # wandb logging
    wandb_log = False  # disabled by default
    wandb_project = "llamac"
    wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # data
    batch_size = 32  # if gradient_accumulation_steps > 1, this is the micro-batch size
    max_seq_len = 256
    vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
    vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens
    # model
    dim = 288
    n_layers = 6
    n_heads = 6
    n_kv_heads = 6
    multiple_of = 32
    dropout = 0.0
    # adamw optimizer
    gradient_accumulation_steps = 4  # used to simulate larger batch sizes
    learning_rate = 5e-4  # max learning rate
    max_iters = 1000  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 10  # how many steps to warm up for
    # system
    device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = "float32"  # float32|bfloat16|float16
    compile = False  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    exec(open("configurator.py").read())  # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    # fixing some hyperparams to sensible defaults
    lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
    min_lr = 5e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # validating checks
    assert vocab_source in ["llama2", "custom"]
    assert vocab_source == "custom" or vocab_size == 32000, "The vocab from Meta has 32K tokens"

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
    if master_process:
        print(f"tokens per iteration will be: {tokens_per_iter:,}")
        print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=torch.float32)
    )

    # task-specific setup
    iter_batches = partial(
        Task.iter_batches,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        device=device,
        num_workers=0,
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 4.0

    # model init
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )  # start with model_args from command line
    if init_from == "scratch":
        raise NotImplementedError
    elif init_from == "resume":
        print(f"Resuming training from {starting_checkpoint}")
        # resume training from a checkpoint.
        checkpoint = torch.load(starting_checkpoint, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float32))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == "resume" and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
        # construction time since NCCL does not support `ComplexFloat`
        prefix = "_orig_mod." if compile else ""
        model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
        model = DDP(model, device_ids=[ddp_local_rank])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            batch_iter = iter_batches(split=split)
            losses = torch.zeros(eval_iters)  # keep on CPU
            for k in range(eval_iters):
                X, Y = next(batch_iter)
                with ctx:
                    logits = model(X, Y)
                    loss = raw_model.last_loss
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # training loop
    train_batch_iter = iter_batches(split="train")
    X, Y = next(train_batch_iter)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                try:
                    wandb.log(
                        {
                            "iter": iter_num,
                            "tokens": iter_num * tokens_per_iter,
                            "loss/train": losses["train"],
                            "loss/val": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        }, step = iter_num
                    )
                except Exception as e:
                    print(f"logging to wandb failed: {e}")
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    checkpoint = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
                    model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
                loss = loss / gradient_accumulation_steps
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            try:
                X, Y = next(train_batch_iter)
            except StopIteration:
                train_batch_iter = iter_batches(split="train")
                X, Y = next(train_batch_iter)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(
                f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
            )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters + 164000:

            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
            model_export(raw_model, os.path.join(out_dir, "model.bin"), version=0)

            break

    if ddp:
        destroy_process_group()

if __name__ == "__main__":

    tokenized_filenames=["imdb_tokenized_train.bin", "imdb_tokenized_val.bin", "imdb_tokenized_test.bin"]
    files_exist = np.array([os.path.exists(f) for f in tokenized_filenames]).all()
    if files_exist:
        print(f"Found {tokenized_filenames}, skipping tokenization.")
    else:
        process_data()

    train_model()