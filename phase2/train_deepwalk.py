import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from phase2.train import BFloat16Adam, get_vram_stats, _format_time

def train_deepwalk_infonce(
    model,
    pairs_dict,
    num_nodes,
    epochs=10,
    batch_size=32768,
    lr=0.01,
    num_negatives=10,
    temperature=0.1,
    use_amp=True,
    optimizer_type="adam",
    device="cuda",
    checkpoint_prefix="data/deepwalk_checkpoint"
):
    """Training loop for DeepWalk baseline models using temperature-scaled InfoNCE loss."""
    model_name = f"DeepWalk-{model.dim}{'-Norm' if model.normalize else ''}"
    print(f"\n[{model_name} Train] Starting training ({epochs} epochs, lr={lr}, temp={temperature}, batch_size={batch_size:,}, opt={optimizer_type})...")
    t0_start = time.time()
    
    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
    model.to(device)
    model.train()
    
    if optimizer_type.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr*5.0)
    else:
        optimizer = BFloat16Adam(model.parameters(), lr=lr)
        
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and is_cuda))
    
    anchors = pairs_dict['anchors']
    positives = pairs_dict['positives']
    num_pairs = len(anchors)
    
    if is_cuda and not anchors.is_pinned():
        anchors = anchors.pin_memory()
        positives = positives.pin_memory()
        
    epoch_times = []
    
    for epoch in range(epochs):
        t_ep_start = time.time()
        perm = torch.randperm(num_pairs)
        total_loss = 0.0
        num_batches = (num_pairs + batch_size - 1) // batch_size
        samples_processed = 0
        
        with tqdm(total=num_batches, desc=f"{model_name} Ep {epoch+1}/{epochs}") as pbar:
            for b in range(0, num_pairs, batch_size):
                b_idx = perm[b:b + batch_size]
                a_ids = anchors[b_idx].to(device, non_blocking=True)
                p_ids = positives[b_idx].to(device, non_blocking=True)
                curr_batch = len(a_ids)
                samples_processed += curr_batch
                
                n_ids = torch.randint(0, num_nodes, (curr_batch, num_negatives), device=device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=(use_amp and is_cuda), dtype=torch.bfloat16):
                    a_coords = model(a_ids) # [B, dim]
                    p_coords = model(p_ids) # [B, dim]
                    n_coords = model(n_ids) # [B, K, dim]
                    
                    pos_sim = model.compute_cosine_sim(a_coords, p_coords).unsqueeze(1) # [B, 1]
                    neg_sim = model.compute_cosine_sim(a_coords, n_coords) # [B, K]
                    
                    logits = torch.cat([pos_sim, neg_sim], dim=1) / temperature # [B, 1 + K]
                    targets = torch.zeros(curr_batch, dtype=torch.long, device=device)
                    
                    loss = F.cross_entropy(logits, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                pbar.update(1)
                
                if is_cuda and (b % max(1, (batch_size * 5)) == 0):
                    alloc_gb, res_gb, _ = get_vram_stats(device)
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        vram=f"{alloc_gb:.2f}/{res_gb:.2f}GB"
                    )
                else:
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

        ep_duration = time.time() - t_ep_start
        epoch_times.append(ep_duration)
        avg_ep_time = sum(epoch_times) / len(epoch_times)
        rem_epochs = epochs - (epoch + 1)
        eta_sec = rem_epochs * avg_ep_time
        throughput = samples_processed / ep_duration
        avg_loss = total_loss / num_batches
        
        alloc_gb, res_gb, max_alloc_gb = get_vram_stats(device)
        print(
            f"Epoch {epoch+1}/{epochs} done in {_format_time(ep_duration)} "
            f"(Total: {_format_time(time.time() - t0_start)}, ETA: {_format_time(eta_sec)}) | "
            f"Loss: {avg_loss:.4f} | Throughput: {throughput/1e3:.1f}k samples/s | "
            f"VRAM: {alloc_gb:.2f}GB alloc / {res_gb:.2f}GB res (Peak: {max_alloc_gb:.2f}GB)"
        )
        
        ckpt_path = f"{checkpoint_prefix}_e{epoch+1}.pt"
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'weight': model.emb.weight.detach().cpu(),
            'num_nodes': num_nodes,
            'dim': model.dim,
            'normalize': model.normalize,
            'loss': avg_loss,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"  Checkpoint saved to {ckpt_path}")

    print(f"[{model_name} Train] Training completed in {_format_time(time.time() - t0_start)}.")
