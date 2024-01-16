import os
import torch
from torch2trt import torch2trt
from torch2trt import TRTModule
import torch.nn as nn
from torchvision.models import resnet18
import time, tracemalloc
import PIL.Image as Image
import torchvision.transforms as T

model_path = 'resnet.pth'
images_path = 'images'

with open("imagenet_classes.txt", "r") as f:
    labels = [line.strip() for line in f]

convert_tensor = T.ToTensor()
transforms = nn.Sequential(
            T.Resize([256, ]),  # We use single int value inside a list due to torchscript type restrictions
            T.CenterCrop(224),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

def get_image_paths(directory):
    image_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        image_paths.append(file_path)
    return image_paths

def transform_image(image): 
    tensor_image = convert_tensor(image)
    transformed_image = transforms(tensor_image)
    return torch.unsqueeze(transformed_image, 0).cuda()


def print_stats(avg_time):
    print("Avg time: {:.4f} seconds".format(avg_time))
    used_memory = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
    print('Used memory: {:.2f} MB'.format(used_memory))

def print_result(image_name, output):
    predicted_class_index = output.argmax(dim=1).item()
    print("Predicted result for {} : {}".format(image_name, labels[predicted_class_index]))

image_paths = get_image_paths(images_path)

images = {}

for image_path in image_paths:
    image = Image.open(image_path)
    transformed_image = transform_image(image)
    images[image_path] = transformed_image

print("Running resnet18")
tracemalloc.start()

resnet = resnet18(pretrained=True).eval().cuda()

start = time.time()
for path, image in images.items():
    output = resnet(image)
    print_result(path, output)
end = time.time()

avg_time = (end - start) / len(images)
print_stats(avg_time)

tracemalloc.stop()

print("\nRunning resnet18 optimized by TensorRT")
tracemalloc.start()

if not os.path.exists(model_path):
    path, image = next(iter(images.items()))
    resnet_trt = torch2trt(resnet, [image], max_workspace_size=1<<30, use_onnx=False)
    torch.save(resnet_trt.state_dict(), model_path)
else:
    resnet_trt = TRTModule()
    resnet_trt.load_state_dict(torch.load(model_path))

start = time.time()
for path, image in images.items():
    output = resnet_trt(image)
    print_result(path, output)
end = time.time()

avg_time = (end - start) / len(images)
print_stats(avg_time)

tracemalloc.stop()
