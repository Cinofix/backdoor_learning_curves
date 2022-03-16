import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-pair", type=str, help="Dataset CIFAR, Imagenette, Imagenette160", required=True,
)
parser.add_argument(
    "-clf",
    type=str,
    help="alexnet, vgg16, mobilenetv3, resnet18, resnet34, ResNet-50",
    required=True,
)
parser.add_argument(
    "-trigger_type",
    default="badnet",
    type=str,
    help="Define trigger type. [badnet, invisible]",
)
parser.add_argument(
    "-trigger_size", type=int, help="Size for the BadNet trigger",
)
parser.add_argument(
    "-ppoison", default=0.1, type=float, help="Percentage of poisoning",
)
parser.add_argument("--save_results", action="store_true")
input_args = parser.parse_args()
