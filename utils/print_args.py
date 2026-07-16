"""Readable command-line configuration output."""


def _section(title, rows):
    print(f'\033[1m{title}\033[0m')
    for label, value in rows:
        print(f'  {label:<24}{value}')
    print()


def print_args(args):
    _section(
        'Experiment',
        [
            ('Task', args.task_name),
            ('Model ID', args.model_id),
            ('Training', bool(args.is_training)),
            ('Seed', args.seed),
        ],
    )
    _section(
        'Data',
        [
            ('Dataset', args.data),
            ('Data file', args.data_path),
            ('Features', args.features),
            ('Input / prediction', f'{args.seq_len} / {args.pred_len}'),
            ('Channels', f'{args.enc_in} -> {args.c_out}'),
        ],
    )
    _section(
        'H2-CAST',
        [
            ('Model / FFN dimension', f'{args.d_model} / {args.d_ff}'),
            ('Encoder / decoder layers', f'{args.e_layers} / {args.d_layers}'),
            ('Attention heads', args.n_heads),
            ('Patch length / stride', f'{args.patch_len} / {args.stride or args.patch_len}'),
            ('Candidate graph', args.candidate_graph),
            ('Tau / Top-K', f'{args.tau} / {args.knn_k}'),
            ('Pruning lambda', args.prune_lambda),
            ('Ablation', args.ablation),
        ],
    )
    _section(
        'Optimization',
        [
            ('Epochs / patience', f'{args.train_epochs} / {args.patience}'),
            ('Batch size', args.batch_size),
            ('Learning rate', args.learning_rate),
            ('LR schedule', args.lradj),
            ('Device', f'cuda:{args.gpu}' if args.use_gpu else 'cpu'),
        ],
    )
