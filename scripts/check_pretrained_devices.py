"""Real-artifact FP32 CPU/CUDA consistency and isolated batch-one timing."""
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import psutil
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from visage.model_registry import METHODS, ARTIFACTS
from visage.data import load_manifest
from visage.pretrained_data import pixels
from visage.pretrained_inference import PretrainedPredictor
from visage.runtime import configure_caches, seed_everything, sha256_file, write_json, utc_now
configure_caches(ROOT)


def main():
    seed_everything(42, threads=4)
    if not torch.cuda.is_available():
        raise RuntimeError('The declared CUDA acceptance check requires the available RTX GPU.')
    validation = [r for r in load_manifest(ROOT / 'data/processed/manifest.csv') if r['split'] == 'validation']
    rows = [r for label in range(3) for r in [row for row in validation if row['label'] == label][:4]]
    arrays = []
    for row in rows:
        source = ROOT / 'data/raw/UTKFace' / row['relative_path']
        if sha256_file(source) != row['file_sha256']:
            raise ValueError('Functional validation fixture changed.')
        with Image.open(source) as image:
            arrays.append(pixels(image))
    methods = {}
    process = psutil.Process()
    for method in METHODS:
        before = time.perf_counter()
        cpu = PretrainedPredictor(method, ARTIFACTS / method, 'cpu')
        cpu_load_seconds = time.perf_counter() - before
        expected = np.concatenate([cpu.predict_batch([array]) for array in arrays])
        cpu_rss = process.memory_info().rss
        del cpu; gc.collect()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        before = time.perf_counter()
        gpu = PretrainedPredictor(method, ARTIFACTS / method, 'cuda')
        torch.cuda.synchronize()
        cuda_load_seconds = time.perf_counter() - before
        actual = np.concatenate([gpu.predict_batch([array]) for array in arrays])
        maximum = float(np.max(np.abs(expected - actual)))
        tolerance = bool(np.allclose(expected, actual, atol=2e-4, rtol=1e-3))
        disagreements = np.flatnonzero(expected.argmax(1) != actual.argmax(1)).tolist()
        if not tolerance or disagreements:
            raise AssertionError({'method': method, 'max_absolute_difference': maximum,
                'labels_disagree': disagreements, 'within_tolerance': tolerance})
        for _ in range(5):
            gpu.predict_batch([arrays[0]])
        measurements = []
        for i in range(30):
            torch.cuda.synchronize(); started = time.perf_counter()
            gpu.predict_batch([arrays[i % len(arrays)]])
            torch.cuda.synchronize()
            measurements.append((time.perf_counter() - started) * 1000)
        methods[method] = {'labels_match': True, 'within_score_tolerance': tolerance,
            'maximum_absolute_score_difference': maximum, 'cpu_load_seconds': cpu_load_seconds,
            'cuda_load_seconds': cuda_load_seconds, 'cpu_process_rss_bytes_after_loading_one_model': cpu_rss,
            'artifact_bytes': (ARTIFACTS / method / 'best.pt').stat().st_size,
            'artifact_sha256': gpu.artifact_sha256, 'cpu_scores': expected.tolist(), 'cuda_scores': actual.tolist(),
            'cuda_batch_one_median_ms': float(np.median(measurements)), 'cuda_batch_one_p95_ms': float(np.percentile(measurements, 95)),
            'cuda_batch_one_timings_ms': measurements,
            'cuda_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
            'cuda_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2}
        del gpu; gc.collect(); torch.cuda.empty_cache()
    # Verify the actual three-model GPU service together, not only isolated
    # architecture adapters. Same raw bytes must select exactly the same pixels.
    from visage.service import InferenceService
    pipeline = json.loads((ROOT / 'data/cache/pretrained_pipeline256/metadata.json').read_text())
    accepted_ids = {r['sample_id'] for r in pipeline['outcomes'] if r['status'] == 'accepted'}
    raw_rows = [r for label in range(3) for r in [row for row in validation
        if row['label'] == label and row['sample_id'] in accepted_ids][:2]]
    cpu_service = InferenceService(device='cpu')
    reference = []
    for row in raw_rows:
        data = (ROOT / 'data/raw/UTKFace' / row['relative_path']).read_bytes()
        reference.append(cpu_service.predict(data, method='all'))
    cpu_service.close(); del cpu_service; gc.collect()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    gpu_service = InferenceService(device='cuda')
    raw_checks = []
    for row, expected in zip(raw_rows, reference, strict=True):
        actual = gpu_service.predict((ROOT / 'data/raw/UTKFace' / row['relative_path']).read_bytes(), method='all')
        labels = lambda value: {p['method']: p['class_id'] for p in value['predictions']}
        if labels(expected) != labels(actual) or expected['face_processing']['pixel_sha256'] != actual['face_processing']['pixel_sha256']:
            raise AssertionError('Three-model raw service differs between CPU and CUDA.')
        raw_checks.append({'sample_id': row['sample_id'], 'labels': labels(actual),
            'shared_crop_sha256': actual['face_processing']['pixel_sha256'], 'matches_cpu': True})
    combined_gpu_peak = torch.cuda.max_memory_allocated() / 1024**2
    gpu_service.close(); del gpu_service; gc.collect(); torch.cuda.empty_cache()
    record = {'ok': True, 'checked_utc': utc_now(), 'atol': .0002, 'rtol': .001,
        'numeric_precision': 'FP32 on CPU and CUDA; TF32 disabled', 'torch_threads': 4,
        'fixtures': [{k: row[k] for k in ('sample_id', 'label', 'file_sha256')} for row in rows],
        'scope': 'Twelve fixed validation crops, four per provided age group; functional consistency, not independent accuracy.',
        'timing_scope': 'Warm CUDA batch-one classification with uint8 arrays already decoded/resized; normalization and device transfer included; no concurrent training.',
        'methods': methods, 'raw_service_cpu_cuda': {'checks': raw_checks,
            'three_models_loaded_together': True, 'gpu_peak_allocated_mib': combined_gpu_peak}}
    write_json(ROOT / 'reports/pretrained/device_verification.json', record)
    print(json.dumps({'ok': True, 'methods': list(methods)}), flush=True)


if __name__ == '__main__':
    main()
