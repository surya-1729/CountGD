import torch
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from util.slconfig import SLConfig
from models.registry import MODULE_BUILD_FUNCS
from util.misc import nested_tensor_from_tensor_list
import datasets.transforms as T

# DEVICE
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

# ARGS
def get_args():
    args = type('Args', (), {})()
    args.modelname = 'groundingdino'
    args.pretrain_model_path = 'checkpoints/checkpoint_fsc147_best.pth'
    args.device = get_device()
    return args

# MODEL
def build_model_and_transform(args):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize,
    ])
    
    # Create a wrapper function that only takes image as input
    def transform_image(image):
        return transform(image, None)[0]  # Add None as target and return only image

    
    cfg = SLConfig.fromfile("/home/alla/CountGD/config/cfg_fsc147_test.py")
    cfg.merge_from_dict({"text_encoder_type": "checkpoints/bert-base-uncased"})
    cfg.device = get_device()

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict

    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, _, _ = build_func(cfg)
    checkpoint = torch.load(args.pretrain_model_path, map_location="cpu")["model"]
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model, transform_image

# INFERENCE
def count_objects(image_path, text_prompt, model, transform, device):
    image = Image.open(image_path)
    input_image = transform(image)
    input_image = input_image.unsqueeze(0).to(device)
    
    # Create a dummy exemplar tensor
    dummy_exemplar = torch.zeros((0, 3, 800, 1333), device=device) 
    
    with torch.no_grad():
        model_output = model(
            samples=nested_tensor_from_tensor_list(input_image),
            exemplars=[dummy_exemplar],  # Pass a list containing the dummy tensor
            captions=text_prompt + " .",
            labels=None
        )
    
    logits = model_output["pred_logits"].sigmoid()[0]
    boxes = model_output["pred_boxes"][0]
    
    conf_thresh = 0.23
    box_mask = logits.max(dim=-1).values > conf_thresh
    logits = logits[box_mask, :].cpu().numpy()
    boxes = boxes[box_mask, :].cpu().numpy()
    
    return boxes, image

# VISUALIZATION
def visualize_results(image, boxes):
    w, h = image.size
    det_map = np.zeros((h, w))
    det_map[(h * boxes[:, 1]).astype(int), (w * boxes[:, 0]).astype(int)] = 1
    det_map = ndimage.gaussian_filter(det_map, sigma=(w // 200, w // 200), order=0)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(det_map[None, :].transpose(1, 2, 0), 'jet', interpolation='none', alpha=0.7)
    plt.title(f'Detected Turkeys: {boxes.shape[0]}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('detection_results.png')  # Save figure instead of showing it
    plt.close()  # Close the figure to free memory


if __name__ == "__main__":
    args = get_args()
    model, transform = build_model_and_transform(args)
    model = model.to(args.device)
    
    image_path = "/home/alla/CountGD/data/data_count/2023-11-19-22_10_30.jpg"
    text_prompt = "bird"
    
    boxes, image = count_objects(image_path, text_prompt, model, transform, args.device)
    visualize_results(image, boxes)
    print(f"Total objects counted: {boxes.shape[0]}")
