import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class BFloat16Adam(torch.optim.Optimizer):
    """Adam optimizer storing m and v states in bfloat16 to fit 128D embeddings in 8GB VRAM."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.to(torch.bfloat16)
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, dtype=torch.bfloat16)
                    
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step = state['step'] + 1
                state['step'] = step
                
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                
                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)
                step_size = lr / bias_correction1
                
                denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
        return loss

def _format_time(seconds):
    """Format seconds into HH:MM:SS or MM:SS."""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes:02d}m {secs:02d}s"

def get_vram_stats(device="cuda"):
    """Get current CUDA VRAM statistics in GB."""
    if not torch.cuda.is_available() or device == "cpu":
        return 0.0, 0.0, 0.0
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    max_allocated = torch.cuda.max_memory_allocated(device) / 1e9
    return allocated, reserved, max_allocated

def train_warmstart(
    model,
    edge_index,
    num_nodes,
    epochs=1,
    batch_size=131072,
    lr=0.01,
    num_negatives=5,
    use_amp=True,
    device="cuda"
):
    """Spring / force-directed warm-start pretraining on the flat torus manifold (Fast SGD)."""
    print(f"\n[Phase2 Warm-Start] Starting spring pretraining ({epochs} epochs, lr={lr}, batch_size={batch_size:,}, opt=SGD)...")
    t0_start = time.time()
    
    is_cuda = torch.cuda.is_available() and device.startswith("cuda")
    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
    model.to(device)
    model.train()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=lr*5.0)
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and is_cuda))
    
    num_edges = edge_index.shape[1]
    src_all = edge_index[0]
    dst_all = edge_index[1]
    
    epoch_times = []
    
    for epoch in range(epochs):
        t_ep_start = time.time()
        perm = torch.randperm(num_edges)
        total_loss = 0.0
        num_batches = (num_edges + batch_size - 1) // batch_size
        samples_processed = 0
        
        with tqdm(total=num_batches, desc=f"Warm-Start Ep {epoch+1}/{epochs}") as pbar:
            for b in range(0, num_edges, batch_size):
                b_idx = perm[b:b + batch_size]
                heads = src_all[b_idx].to(device, non_blocking=True)
                tails = dst_all[b_idx].to(device, non_blocking=True)
                
                curr_batch = len(heads)
                samples_processed += curr_batch
                
                neg_tails = torch.randint(0, num_nodes, (curr_batch, num_negatives), device=device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=(use_amp and is_cuda), dtype=torch.bfloat16):
                    h_coords = model(heads) # [B, 128]
                    t_coords = model(tails) # [B, 128]
                    n_coords = model(neg_tails) # [B, K, 128]
                    
                    pos_dot = model.compute_dot(h_coords, t_coords) # [B]
                    loss_attract = (float(model.dim / 2) - pos_dot).mean()
                    
                    neg_dot = model.compute_dot(h_coords, n_coords) # [B, K]
                    loss_repel = torch.relu(neg_dot).mean()
                    
                    loss = loss_attract + loss_repel

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                pbar.update(1)
                
                if is_cuda and (b % max(1, (batch_size * 2)) == 0):
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
        
        alloc_gb, res_gb, max_alloc_gb = get_vram_stats(device)
        print(
            f"Epoch {epoch+1}/{epochs} done in {_format_time(ep_duration)} "
            f"(Total: {_format_time(time.time() - t0_start)}, ETA: {_format_time(eta_sec)}) | "
            f"Loss: {total_loss/num_batches:.4f} | Throughput: {throughput/1e3:.1f}k samples/s | "
            f"VRAM: {alloc_gb:.2f}GB alloc / {res_gb:.2f}GB res (Peak: {max_alloc_gb:.2f}GB)"
        )

    print(f"[Phase2 Warm-Start] Completed in {_format_time(time.time() - t0_start)}.")


def train_infonce(
    model,
    pairs_dict,
    num_nodes,
    epochs=10,
    batch_size=16384,
    lr=0.01,
    num_negatives=10,
    temperature=0.1,
    use_amp=True,
    optimizer_type="adam",
    device="cuda",
    checkpoint_prefix="data/phase2_checkpoint"
):
    """Main training loop using temperature-scaled InfoNCE loss over walk co-occurrences."""
    print(f"\n[Phase2 InfoNCE Train] Starting main training ({epochs} epochs, lr={lr}, temp={temperature}, batch_size={batch_size:,}, opt={optimizer_type})...")
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
        
        with tqdm(total=num_batches, desc=f"InfoNCE Ep {epoch+1}/{epochs}") as pbar:
            for b in range(0, num_pairs, batch_size):
                b_idx = perm[b:b + batch_size]
                a_ids = anchors[b_idx].to(device, non_blocking=True)
                p_ids = positives[b_idx].to(device, non_blocking=True)
                curr_batch = len(a_ids)
                samples_processed += curr_batch
                
                n_ids = torch.randint(0, num_nodes, (curr_batch, num_negatives), device=device)
                
                optimizer.zero_grad(set_to_none=True)
                
                with torch.amp.autocast('cuda', enabled=(use_amp and is_cuda), dtype=torch.bfloat16):
                    a_coords = model(a_ids) # [B, 128]
                    p_coords = model(p_ids) # [B, 128]
                    n_coords = model(n_ids) # [B, K, 128]
                    
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
                
                if is_cuda and (b % (batch_size * 10) == 0):
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
            'theta': model.theta.detach().cpu(),
            'num_nodes': num_nodes,
            'num_angles': model.num_angles,
            'dim': model.dim,
            'loss': avg_loss,
        }
        torch.save(checkpoint, ckpt_path)
        print(f"  Checkpoint saved to {ckpt_path}")

    print(f"[Phase2 InfoNCE Train] Main training completed in {_format_time(time.time() - t0_start)}.")
