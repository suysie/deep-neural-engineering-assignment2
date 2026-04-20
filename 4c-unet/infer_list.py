import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import argparse
from model import UNet
import numpy as np

# load a grayscale model
def load_model(model_path, device='cpu'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = UNet(in_channels=1, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"Model loaded successfully from {model_path}")
    return model

def preprocess_image(image_path, target_size=(256, 256)):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")
    img = Image.open(image_path).convert('L')  # Convert to Grayscale
#    img = Image.open(image_path).convert('RGB')  # Convert to Grayscale
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return tensor, img.size  # Return tensor and original size for resizing back later

def infer(model, input_tensor, device, threshold=0.5):
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        output = torch.sigmoid(output)
        binary_output = (output > threshold).float()
        return binary_output

def postprocess_and_save(tensor, original_size, output_path):
    img_tensor = tensor.squeeze(0).squeeze(0)
    img_np = img_tensor.cpu().numpy()
    img_np = (img_np * 255).astype('uint8')
    img_pil = Image.fromarray(img_np)
    img_resized = img_pil.resize(original_size, Image.BILINEAR)
    img_resized.save(output_path)
    print(f"Result saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Infer connected characters from dotted images.")
    parser.add_argument("--input", type=str, default="./data/test_input")
    parser.add_argument("--output", type=str, default="./output/test_epoch3/", help="Path to save the output image")
    parser.add_argument("--model", type=str, default="checkpoints/test_dataset3/model_epoch_3_dice.pth", help="Path to the trained model")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256],
                        help="Target size for model input (width height)")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        # 1. Load Model
        model = load_model(args.model, device)
        # Supported image extensions, only png used in testing
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        # Get list of files
        files = [f for f in os.listdir(args.input) if os.path.isfile(os.path.join(args.input, f))]
        image_files = [f for f in files if os.path.splitext(f)[1].lower() in valid_extensions]
        if not image_files:
            print(f"No supported image files found in {args.input}")
            return
        print(f"Found {len(image_files)} images to infer.")
        for filename in image_files:
            input_path = os.path.join(args.input, filename)
            output_path = os.path.join(args.output, filename)
            # 2. Preprocess Input
            input_tensor, orig_size = preprocess_image(input_path, target_size=tuple(args.size))
            print(f"Processing image: {input_path} (Original size: {orig_size})")
            # 3. Infer
            result_tensor = infer(model, input_tensor, device)
            # 4. Save Result
            postprocess_and_save(result_tensor, orig_size, output_path)
    except Exception as e:
        print(f"Error during inference: {e}")


if __name__ == "__main__":
    main()