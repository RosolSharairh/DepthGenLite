import warnings
import os
import torch
import lpips
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from skimage.transform import resize
import numpy as np
from datasets.dataset import ImageDataset  # Import the custom dataset class

warnings.filterwarnings("ignore")

# Initialize LPIPS with correct input range [-1, 1]
lpips_fn = lpips.LPIPS(net='alex')
device = 'cuda'
lpips_fn.to(device)

# Directories
ground_truth_dir = "C:\\Users\\Orange\\GradProject_\\GradProjectData\\celebaa\\celeba_test"
output_dir = "C:\\Users\\Orange\\GradProject_\\GradProjectData\\Outputs\\lightweight"
mask_dir = "C:\\Users\\Orange\\GradProject_\\GradProjectData\\img_mask\\img_mask_test"  # Corrected directory

transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to a consistent size
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1,1]
])

# Metrics dictionaries
metrics = {
    '0-20': {'lpips': [], 'psnr': [], 'ssim': []},
    '20-40': {'lpips': [], 'psnr': [], 'ssim': []},
    '40-60': {'lpips': [], 'psnr': [], 'ssim': []},
}

def calculate_mask_ratio(mask):
    mask_area = torch.sum(mask == 0).item()  # Assuming 0 represents the mask
    total_area = mask.shape[1] * mask.shape[2]  # mask is a tensor, so need to use shape[1] and shape[2]
    return (mask_area / total_area) * 100

# Create dataset instance
image_dataset = ImageDataset(
    image_root=ground_truth_dir,
    mask_root=mask_dir,
    load_size=(256, 256),  # Consistent load size
    mode='test'
)

# Process each image
for i in range(len(image_dataset)):
    image, mask, edge, gray_image, image_name = image_dataset[i]

    opt_path = os.path.join(output_dir, image_name)
    if not os.path.exists(opt_path):
        print(f"Output not found for {image_name}, skipping.")
        continue

    gt_image = io.imread(os.path.join(ground_truth_dir, image_name))
    opt_image = io.imread(opt_path)

    if gt_image.shape != opt_image.shape:
        opt_image = resize(opt_image, gt_image.shape, anti_aliasing=True, mode='reflect', preserve_range=True)
        opt_image = opt_image.astype(gt_image.dtype)

    gt_img_for_lpips = transform(Image.open(os.path.join(ground_truth_dir, image_name)).convert('RGB')).unsqueeze(0).to(device)
    opt_img_for_lpips = transform(Image.open(opt_path).convert('RGB')).unsqueeze(0).to(device)

    # Calculate LPIPS distance
    with torch.no_grad():
        d = lpips_fn(opt_img_for_lpips, gt_img_for_lpips).item()

    # Calculate PSNR
    current_psnr = psnr(gt_image, opt_image, data_range=gt_image.max() - gt_image.min())

    # Determine appropriate win_size for SSIM
    win_size = min(gt_image.shape[0], gt_image.shape[1], 7)
    if win_size % 2 == 0:
        win_size -= 1

    # Calculate SSIM
    ssim_value = ssim(gt_image, opt_image, win_size=win_size, data_range=gt_image.max() - gt_image.min(), channel_axis=-1)

    # Calculate mask ratio
    mask_ratio = calculate_mask_ratio(mask)

    # Classify based on mask ratio and store metrics
    if 0 <= mask_ratio < 20:
        category = '0-20'
    elif 20 <= mask_ratio < 40:
        category = '20-40'
    elif 40 <= mask_ratio < 60:
        category = '40-60'
    else:
        continue  # Skip masks outside the 0-60% range

    metrics[category]['lpips'].append(d)
    metrics[category]['psnr'].append(current_psnr)
    metrics[category]['ssim'].append(ssim_value)

# Calculate and print averages for each mask ratio category
for category, values in metrics.items():
    avg_lpips = sum(values['lpips']) / len(values['lpips']) if values['lpips'] else 0
    avg_psnr = sum(values['psnr']) / len(values['psnr']) if values['psnr'] else 0
    avg_ssim = sum(values['ssim']) / len(values['ssim']) if values['ssim'] else 0
    print(f"Mask Ratio {category}% - Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}, Average LPIPS: {avg_lpips}")
