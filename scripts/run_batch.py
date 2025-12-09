from typing import List, Dict
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import cpu_count

from nisqab.core.model_torch import model_init
from nisqab.utils.file_utils import yamlparser
from nisqab.utils.dataset import NISQADataset, collate_fn
from nisqab.utils.audio_cache import create_audio_length_cache
from nisqab.utils.audio_sampler import LengthBasedBatchSampler


def batch_inference(
    model,
    file_paths,
    audio_lengths,
    device="cuda",
    batch_size=32,
    dtype=torch.bfloat16,
    num_workers=8
    ) -> List[Dict]:

    # Move model to device
    model = model.to(device)
    model = model.to(dtype)
    model.eval()
    
    # Create dataset
    dataset = NISQADataset(
        file_paths=file_paths,
        audio_lengths=audio_lengths,
        seg_length=15,
        seg_hop=1,
        max_length=2000,
        n_fft=4096,
        hop_length=0.01,
        win_length=0.02,
        n_mels=48,
        fmax=20000
    )
    sampler = LengthBasedBatchSampler(
        file_paths=file_paths,
        audio_lengths=audio_lengths,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=num_workers > 0
    )
    
    results = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Inference", total=len(dataloader))):
            x_batch = batch['x_spec_seg'].to(device).to(dtype)
            n_wins_batch = batch['n_wins'].to(device)
  
            if dtype == torch.bfloat16 and device == "cuda":
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(x_batch, n_wins_batch)
            else:
                outputs = model(x_batch, n_wins_batch)
            
            for i in range(len(batch['file_paths'])):
                MOS, NOI, DISC, COL, LOUD = outputs[i].cpu().tolist()
                results.append({
                    'file': batch['file_paths'][i],
                    'MOS': MOS,
                    'NOI': NOI,
                    'DISC': DISC,
                    'COL': COL,
                    'LOUD': LOUD
                })
    
    return results


if __name__ == "__main__":
    args = yamlparser()
    print(args)
    with open(args["yaml"], "r") as ymlfile:
        args_yaml = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = {**args_yaml, **args}
    model = model_init(args)

    file_paths = [
        r"C:\Users\nicit\Downloads\audio_7000_82cd76cba6e7a76a6ae5.wav",
    ]*32
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    
    cache_file = "./audio_lengths_cache.json"
    audio_lengths = create_audio_length_cache(
        file_paths=file_paths,
        cache_file=cache_file,
        num_workers=min(4, cpu_count()),
        force_rebuild=False
    )
    
    results = batch_inference(
        model=model,
        file_paths=file_paths,
        audio_lengths=audio_lengths,
        device=device,
        batch_size=batch_size,
        dtype=torch.float32,
        num_workers=min(4, cpu_count()),
    )
    print(results[0])

