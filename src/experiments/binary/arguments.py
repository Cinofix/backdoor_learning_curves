import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "-pair",
    type=str,
    help="Dataset pair. MNIST {7-1, 3-0, 5-2} CIFAR {6-0, 0-9, 2-5}",
    required=True,
)
parser.add_argument(
    "-clf", type=str, help="svm, svm-rbf, logistic, ridge", required=True
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
