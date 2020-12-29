import sys
import getopt
import time
import torch
import csv
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision import transforms
from torchvision.models.alexnet import alexnet
from torch2trt import torch2trt
from torch2trt import TRTModule


# Create dictionary with classes
with open('./config/classes.csv', 'r') as fd:
    dc = csv.DictReader(fd)
    classes = {}
    for line in dc:
        classes[int(line['class_id'])] = line['class_name']


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])

Path("./output").mkdir(exist_ok=True)


def process_images(images: list,
                   trt: bool):
    if trt:
        timest = time.time()
        # x1 = torch.ones((1, 3, 224, 224)).cuda()
        # model = alexnet(pretrained=True).eval().cuda()
        # model_trt = torch2trt(model, [x1])
        # torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
        # exit()
        model = TRTModule()
        model.load_state_dict(torch.load('alexnet_trt.pth'))
        print("Model load time {}".format(time.time() - timest))
    else:
        timest = time.time()
        model = alexnet(pretrained=True).eval().cuda()
        print("Model load time {}".format(time.time() - timest))

    for image in images:
        index = classify_image(image, model)
        output_text = str(index) + ': ' + classes[index]
        edit = ImageDraw.Draw(image)
        edit.rectangle((0, image.height - 20, image.width, image.height),
                       fill=(255, 255, 255))
        edit.text((50, image.height-15), output_text, (0, 0, 0),
                  font=ImageFont.load_default())
        image.save('./output/' + image.filename.split('/')[-1])

    print('Memory allocated: ' + str(torch.cuda.memory_allocated()))
    print('Max memory allocated: ' + str(torch.cuda.max_memory_allocated()))


def classify_image(image: Image,
                   model) -> int:
    image_tensor = transform(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input = Variable(image_tensor).to(device)

    # Predict image class
    timest = time.time()
    output = model(input)
    print("Image processing time {}".format(time.time() - timest))
    return output.data.cpu().numpy().argmax()


def print_usage():
    print("Usage: python lab3.py [--trt] <image> [image...]")


def main(argv: list,
         trt: bool = False):
    # Chek arguments and enable TensorRT if True
    try:
        opts, _ = getopt.getopt(argv, "", ["trt"])
        if len(opts) == 1:
            trt = True
            argv.remove('--trt')
        elif len(opts) > 1:
            raise getopt.GetoptError("invalid arguments")
    except getopt.GetoptError:
        print_usage()
        sys.exit(1)

    # Open images
    images = []
    for img_path in argv:
        try:
            image = Image.open(img_path)
            images.append(image)
        except FileNotFoundError:
            print(img_path + " not found")

    # Exit with error if there is no images
    if len(images) == 0:
        print_usage()
        sys.exit(1)

    process_images(images, trt)


if __name__ == "__main__":
    main(sys.argv[1:])
