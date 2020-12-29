import sys
import getopt
import time
import torch
from PIL import Image
from torchvision.models.alexnet import alexnet
from torch2trt import torch2trt
from torch2trt import TRTModule


def process_images(images: list,
                   trt: bool):
    if trt:
        timest = time.time()
        x1 = torch.ones((1, 3, 224, 224)).cuda()
        model = alexnet(pretrained=True).eval().cuda()
        model_trt = torch2trt(model, [x1])
        torch.save(model_trt.state_dict(), 'alexnet_trt.pth')
        # exit()
        # model = TRTModule()
        # model.load_state_dict(torch.load('alexnet_trt.pth'))
        print("load time {}".format(time.time() - timest))
    else:
        timest = time.time()
        model = alexnet(pretrained=True).eval().cuda()
        print("load time {}".format(time.time() - timest))


def print_usage():
    """
    Prints script usage.
    :return: None
    """
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

    # Exit with error if no images provided
    if len(argv) == 0:
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

    process_images(images, trt)


if __name__ == "__main__":
    main(sys.argv[1:])
