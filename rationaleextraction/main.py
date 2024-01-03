import argparse
from models.energy_model.train import train
from models.energy_model.test import test_recall

parser = argparse.ArgumentParser(description='jigsaw')
parser.add_argument('--save_path', type=str, default='./logs/rationaleextraction_selection0.3')

parser.add_argument('--num_iterations', type=int, default=-30)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--eval_batch_size', type=int, default=512)
parser.add_argument('--embed_size', type=int, default=300)
parser.add_argument('--proj_size', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=150)

parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--min_lr', type=float, default=1e-5)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--threshold', type=float, default=1e-4)
parser.add_argument('--cooldown', type=int, default=5)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--max_grad_norm', type=float, default=5.)

parser.add_argument('--print_every', type=int, default=100)
parser.add_argument('--eval_every', type=int, default=-1)
parser.add_argument('--save_every', type=int, default=-1)

parser.add_argument('--layer', choices=["lstm", 'bert'], default="lstm")
parser.add_argument('--train_embed', action='store_false', dest='fix_emb')

# rationale settings for HardKuma model
parser.add_argument('--selection', type=float, default=0.3,
                    help="Target text selection rate for Lagrange.")

# lagrange settings
parser.add_argument('--lagrange_lr', type=float, default=0.01,
                        help="learning rate for lagrange")
parser.add_argument('--lagrange_alpha', type=float, default=0.99,
                    help="alpha for computing the running average")
parser.add_argument('--lambda_init', type=float, default=1e-4,
                    help="initial value for lambda1")
parser.add_argument('--lambda_min', type=float, default=1e-12,
                    help="initial value for lambda_min")
parser.add_argument('--lambda_max', type=float, default=5.,
                    help="initial value for lambda_max")
parser.add_argument('--abs', action='store_true', default=False,
                    help='whether to use abs on (c0 - selection)')
parser.add_argument("--eps", default=1e-6, type=float)
parser.add_argument("--mode", default='train')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument("--llm", default='bert', type=str)
parser.add_argument("--path", default='../data/All/')
parser.add_argument("--num_workers", default=-1, type=int)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument("--pretrained_weights", default=None, type=str)
parser.add_argument("--strategy", type=int, default=1,
                    help='which deterministic strategy to choose')
parser.add_argument("--topk", type=int, default=2)

args = parser.parse_args()
if args.mode == 'train':
    train(args)
elif args.mode == 'test_recall':
    test_recall(args)
else:
    raise NotImplementedError