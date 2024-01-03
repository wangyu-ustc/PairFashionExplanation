import pdb
import argparse
from train import train
from test import test
# from test import savewords
# from test import saveprompts
# from test import get_scores
# from test import elle_test
from test import test_recall

parser = argparse.ArgumentParser(description='Pairwise Fashion Explanation')
parser.add_argument('--save_path', type=str, default='logs/default/')
parser.add_argument('--num_iterations', type=int, default=1000000)
parser.add_argument('--print_every', type=int, default=2000)
parser.add_argument('--eval_every', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--eval_batch_size', type=int, default=512)
parser.add_argument("--path", default='../data/All/')
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--fix_emb', default=False, action='store_true')
parser.add_argument("--llm", type=str, default='bert')
parser.add_argument("--mode", type=str, default='train')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument('--lasso', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--threshold', type=float, default=1e-4)
parser.add_argument('--pretrained_weights', type=str, default=None)
args = parser.parse_args()

if args.save_path == '/dev/null/':
    print("WARNING! Your model will not be saved")

if args.mode == 'train':
    train(args)
elif args.mode == 'test':
    test(args)
elif args.mode == 'visualize':
    visualize(args)
elif args.mode == 'savewords':
    savewords(args)
elif args.mode == 'saveprompts':
    saveprompts(args)
elif args.mode == 'scores':
    get_scores(args)
elif args.mode == 'test_recall':
    test_recall(args)

