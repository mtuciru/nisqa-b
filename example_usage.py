#!/usr/bin/env python3
"""
Пример использования оптимизированного NISQA инференса с кешированием и семплерами.
"""

import os
import time
import torch
from multiprocessing import cpu_count

# Импорты для NISQA
from src.core.model_torch import model_init
from src.utils.train_utils import yamlparser
from src.utils.dataset import NISQADataset, collate_fn
from src.utils.audio_cache import create_audio_length_cache, print_length_distribution
from src.utils.audio_sampler import LengthBasedBatchSampler, SimpleBatchSampler
from torch.utils.data import DataLoader


def run_nisqa_inference(file_paths, model_config="configs/config.yaml", 
                       cache_file="audio_cache.json", batch_size=32):
    """
    Запуск NISQA инференса с оптимизациями.
    
    Args:
        file_paths: список путей к аудио файлам
        model_config: путь к конфигу модели
        cache_file: файл для кеширования длин аудио
        batch_size: размер батча
    """
    
    # 1. Загрузка модели
    print("Loading NISQA model...")
    args = {"yaml": model_config}
    model = model_init(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Создание кеша длин аудио (многопроцессорно)
    print(f"\nCreating audio length cache for {len(file_paths)} files...")
    audio_lengths = create_audio_length_cache(
        file_paths=file_paths,
        cache_file=cache_file,
        num_workers=min(8, cpu_count()),
        force_rebuild=False  # Использовать существующий кеш если есть
    )
    
    # Показать распределение
    print_length_distribution(audio_lengths)
    
    # 3. Создание датасета
    dataset = NISQADataset(
        file_paths=file_paths,
        audio_lengths=audio_lengths
    )
    
    # 4. Создание умного семплера
    sampler = LengthBasedBatchSampler(
        file_paths=file_paths,
        audio_lengths=audio_lengths,
        batch_size=batch_size,
        shuffle=False  # Для воспроизводимости
    )
    
    print(f"\nSampler created:")
    sampler.print_stats()
    
    # 5. Создание DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=min(4, cpu_count()),
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False
    )
    
    # 6. Инференс
    model = model.to(device)
    model.eval()
    
    results = []
    start_time = time.time()
    
    print(f"\nRunning inference...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x_batch = batch['x_spec_seg'].to(device)
            n_wins_batch = batch['n_wins'].to(device)
            
            # Forward pass
            outputs = model(x_batch, n_wins_batch)
            
            # Collect results
            for i in range(len(batch['file_paths'])):
                results.append({
                    'file': batch['file_paths'][i],
                    'mos': outputs[i][0].item(),
                    'noi': outputs[i][1].item(),
                    'dis': outputs[i][2].item(),
                    'col': outputs[i][3].item(),
                    'loud': outputs[i][4].item(),
                    'n_segments': n_wins_batch[i].item(),
                    'audio_length': audio_lengths.get(batch['file_paths'][i], 0.0)
                })
            
            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {batch_idx + 1}/{len(sampler)} batches")
    
    total_time = time.time() - start_time
    
    # 7. Результаты
    print(f"\n=== Results ===")
    print(f"Total files processed: {len(results)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average per file: {total_time / len(results):.4f}s")
    print(f"Throughput: {len(results) / total_time:.2f} files/sec")
    
    return results


def main():
    """Пример использования"""
    
    # Пример файлов (замените на свои)
    file_paths = [
        "/home/nikita/datasets_for_tests/data_cv/000000.wav",
        "/home/nikita/datasets_for_tests/data_cv/000034.wav", 
        "/home/nikita/datasets_for_tests/data_cv/000061.wav",
        "/home/nikita/datasets_for_tests/data_cv/06c40091aedd69700a81f05fb4370aa4.wav"
    ]
    
    # Фильтровать существующие файлы
    existing_files = [f for f in file_paths if os.path.exists(f)]
    
    if not existing_files:
        print("No audio files found! Please update file_paths in the script.")
        return
    
    print(f"Found {len(existing_files)} audio files")
    
    # Запуск инференса
    results = run_nisqa_inference(
        file_paths=existing_files,
        model_config="configs/config.yaml",
        cache_file="example_audio_cache.json",
        batch_size=4  # Маленький батч для примера
    )
    
    # Показать результаты
    print(f"\n=== Sample Results ===")
    for r in results[:3]:
        print(f"File: {os.path.basename(r['file'])}")
        print(f"  MOS: {r['mos']:.3f}")
        print(f"  NOI: {r['noi']:.3f}, DIS: {r['dis']:.3f}")
        print(f"  COL: {r['col']:.3f}, LOUD: {r['loud']:.3f}")
        print(f"  Audio length: {r['audio_length']:.2f}s")
        print(f"  Segments: {r['n_segments']}")
        print()


if __name__ == "__main__":
    main()

