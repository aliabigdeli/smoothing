""" This script loads a base classifier and then runs PREDICT on many examples from a dataset.
"""
import argparse
import setGPU
from datasets import get_dataset, DATASETS, get_num_classes
from core import Smooth
from time import time
import torch
from architectures import get_architecture
import datetime
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser(description='Predict on many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("-o", "--outfile", type=str, default=None, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # Ensure CUDA is available
    if torch.cuda.is_available():
        free_memory = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        print(f"Free CUDA memory: {free_memory / (1024 ** 3):.2f} GB")
    else:
        print("CUDA is not available.")

    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smoothed classifier g
    smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    if args.outfile:
        f = open(args.outfile, 'w')
    else:
        f = open(f'accpred/sigma_{args.sigma}_skip_{args.skip}.txt', 'w')
    print("idx\tlabel\tpredict\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    y_pred = []
    y_pred_sum = []
    y_true = []
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.cuda()
        before_time = time()

        # make the prediction
        # prediction = smoothed_classifier.predict(x, args.N, args.alpha, args.batch)
        # perdiction_sum = smoothed_classifier.predict_sum(x, args.N, args.alpha, args.batch)
        prediction, prediction_sum = smoothed_classifier.predict_both(x, args.N, args.alpha, args.batch)

        after_time = time()
        y_pred.append(prediction)
        y_true.append(label)
        y_pred_sum.append(prediction_sum)
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

        # log the prediction and whether it was correct
        print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction, correct, time_elapsed), file=f, flush=True)

    f.close()

    accuracy = accuracy_score(y_true, y_pred)
    acuuracy_sum = accuracy_score(y_true, y_pred_sum)

    with open(f'accpred/acc_sigma_{args.sigma}_skip_{args.skip}.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy}, Accuracy_sum: {acuuracy_sum}')