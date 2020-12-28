import sys
import getopt
from PIL import Image


def print_usage():
    """
    Prints script usage.
    :return: None
    """
    print("Usage: python lab3.py [--trt] <image> [image...]")


def main(argv: list,
         trt: bool = False):
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

    for img_path in argv:
        try:
            image = Image.open(img_path)
        except FileNotFoundError:
            print(img_path + " not found")


if __name__ == "__main__":
    main(sys.argv[1:])
