import argparse
import random

import numpy as np
import torch

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args


def build_setting(args, iteration):
    """Create a collision-resistant experiment/checkpoint identifier."""

    stride = args.stride if args.stride is not None else args.patch_len
    return (
        f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_'
        f'ft{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_'
        f'dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_'
        f'df{args.d_ff}_pa{args.patch_len}_st{stride}_tau{args.tau}_'
        f'cg{args.candidate_graph}_ab{args.ablation}_{args.des}_{iteration}'
    )


def build_parser():
    parser = argparse.ArgumentParser(description='H2-CAST forecasting experiments')

    # Experiment and data.
    parser.add_argument(
        '--task_name',
        required=True,
        choices=['long_term_forecast', 'short_term_forecast'],
    )
    parser.add_argument('--is_training', type=int, required=True, choices=[0, 1])
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--model', choices=['HHCAST'], default='HHCAST')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--root_path', type=str, default='./data/')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv')
    parser.add_argument('--features', choices=['M', 'S', 'MS'], default='M')
    parser.add_argument('--target', type=str, default='OT')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--embed', choices=['timeF', 'calendar'], default='timeF')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--inverse', action='store_true')

    # Forecasting window.
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)

    # H2-CAST architecture.
    parser.add_argument('--enc_in', type=int, default=7)
    parser.add_argument('--c_out', type=int, default=7)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patch_len', type=int, default=16)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument(
        '--candidate_graph', choices=['threshold', 'topk'], default='threshold'
    )
    parser.add_argument('--knn_k', type=int, default=5)
    parser.add_argument('--prune_lambda', type=float, default=0.05)
    parser.add_argument('--temporal_layers', type=int, default=None)
    parser.add_argument('--decoder_dim', type=int, default=None)
    parser.add_argument('--decoder_heads', type=int, default=None)
    parser.add_argument('--decoder_dropout', type=float, default=None)
    parser.add_argument('--use_revin', type=int, choices=[0, 1], default=0)
    parser.add_argument('--revin_affine', type=int, choices=[0, 1], default=0)
    parser.add_argument('--revin_subtract_last', type=int, choices=[0, 1], default=0)
    parser.add_argument(
        '--ablation',
        choices=[
            'full',
            'no_hhc',
            'no_dcp',
            'global_only',
            'subgroup_only',
            'no_bfd',
        ],
        default='full',
    )

    # Optimization and evaluation.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--des', type=str, default='Exp')
    parser.add_argument(
        '--lradj', choices=['type1', 'type2', 'cosine', 'unchanged'], default='cosine'
    )
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=2021)

    # Device.
    parser.add_argument('--use_gpu', type=int, choices=[0, 1], default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    return parser


def main():
    args = build_parser().parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        args.device_ids = [int(device_id) for device_id in args.devices.split(',')]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.is_training:
        for iteration in range(args.itr):
            setting = build_setting(args, iteration)
            experiment = Exp_Long_Term_Forecast(args)
            print(f'>>>>>>>start training: {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
            experiment.train(setting)
            print(f'>>>>>>>testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            experiment.test(setting)
            torch.cuda.empty_cache()
    else:
        setting = build_setting(args, 0)
        experiment = Exp_Long_Term_Forecast(args)
        print(f'>>>>>>>testing: {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        experiment.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
