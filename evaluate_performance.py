import torch
import time
import psutil
from torch.utils.data import DataLoader
from models.generator.generator import Generator  # Adjust to your actual model class
from datasets.dataset import create_image_dataset  # Adjust to your actual dataset function
from options.test_options import TestOptions
from utils.setSeed import setSeed
from fvcore.nn import FlopCountAnalysis, parameter_count_table

# Set seed for reproducibility
setSeed(42)

# Measure Inference Time
def measure_inference_time(model, dataloader, device):
    model.eval()
    model.to(device)
    total_time = 0.0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= 100:  # Limit to 100 batches for measurement
                break
            ground_truth, mask, edge, gray_image, _ = data
            input_image = ground_truth * mask
            input_edge = edge * mask
            inputs = (input_image, torch.cat((input_edge, gray_image * mask), dim=1), mask)
            inputs = tuple(i.to(device) for i in inputs)
            start_time = time.time()
            outputs = model(*inputs)
            end_time = time.time()
            total_time += (end_time - start_time)
    return total_time / min(len(dataloader), 100)  # Average over measured batches

# Measure GPU Memory Usage
def measure_gpu_memory_usage(model, dataloader, device):
    model.eval()
    model.to(device)
    total_memory = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= 100:  # Limit to 100 batches for measurement
                break
            ground_truth, mask, edge, gray_image, _ = data
            input_image = ground_truth * mask
            input_edge = edge * mask
            inputs = (input_image, torch.cat((input_edge, gray_image * mask), dim=1), mask)
            inputs = tuple(i.to(device) for i in inputs)
            torch.cuda.reset_peak_memory_stats(device)
            outputs = model(*inputs)
            total_memory += torch.cuda.max_memory_allocated(device)
    return total_memory / min(len(dataloader), 100)  # Average over measured batches

# Measure CPU Memory Usage
def measure_cpu_memory_usage(model, dataloader):
    model.eval()
    process = psutil.Process(os.getpid())
    total_memory = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            if i >= 100:  # Limit to 100 batches for measurement
                break
            ground_truth, mask, edge, gray_image, _ = data
            input_image = ground_truth * mask
            input_edge = edge * mask
            inputs = (input_image, torch.cat((input_edge, gray_image * mask), dim=1), mask)
            mem_before = process.memory_info().rss
            outputs = model(*inputs)
            mem_after = process.memory_info().rss
            total_memory += (mem_after - mem_before)
    return total_memory / min(len(dataloader), 100)  # Average over measured batches

# Calculate FLOPs
def calculate_flops(model, input_image, input_edge, mask):
    inputs = (input_image, input_edge, mask)
    flops = FlopCountAnalysis(model, inputs)
    print(f"FLOPs: {flops.total() / 1e9} GFLOPs")
    print(parameter_count_table(model))

def main():
    opts = TestOptions().parse
    print("Parsed options:", opts)

    # Load model
    model_checkpoint = opts.pre_trained
    print("Loading model from:", model_checkpoint)
    model = Generator(image_in_channels=3, edge_in_channels=2, out_channels=3)
    model.load_state_dict(torch.load(model_checkpoint)['generator'])  # Load pre-trained model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    # Load dataset
    dataset = create_image_dataset(opts)
    dataloader = DataLoader(dataset, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)

    # Performance evaluations
    print('start performance evaluation...')
    inference_time = measure_inference_time(model, dataloader, device)
    print(f"Inference Time per Batch: {inference_time:.4f} seconds")

    if device.type == 'cuda':
        gpu_memory_usage = measure_gpu_memory_usage(model, dataloader, device)
        print(f"GPU Memory Usage per Batch: {gpu_memory_usage / 1024**2:.4f} MB")
    else:
        cpu_memory_usage = measure_cpu_memory_usage(model, dataloader)
        print(f"CPU Memory Usage per Batch: {cpu_memory_usage / 1024**2:.4f} MB")

    # Calculate FLOPs
    print("Calculating FLOPs...")
    sample_batch = next(iter(dataloader))
    ground_truth, mask, edge, gray_image, _ = sample_batch
    input_image = ground_truth * mask
    input_edge = edge * mask
    input_image = input_image.to(device)
    input_edge = torch.cat((input_edge, gray_image * mask), dim=1).to(device)
    mask = mask.to(device)
    calculate_flops(model, input_image, input_edge, mask)

if __name__ == '__main__':
    main()
