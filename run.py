import argparse
import os
import torch
from exp.exp_main import Exp_Main
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
def get_parser():
    parser = argparse.ArgumentParser(description='xPatch')

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    # ablation switches  A
    # parser.add_argument('--use_pti', type=str2bool, default=True, help='use position token interaction')
    # parser.add_argument('--use_sdb', type=str2bool, default=False, help='use self-attention based dynamic block')
    # parser.add_argument('--use_ffn', type=str2bool, default=True, help='use feed-forward network')

    # ablation switches  B
    parser.add_argument('--ab_dense',  type=str2bool, default=False, help='Dense graph: disable Top-K (equivalent to K=C)')
    parser.add_argument('--ab_random', type=str2bool, default=False, help='Random-K: random neighbor set with the same K')
    parser.add_argument('--ab_static', type=str2bool, default=False, help='Static-K: use batch-averaged state to build a fixed graph for all samples')

    # ablation switches  C
    # parser.add_argument('--ab_maxpool', type=str2bool, default=False, help='Bottleneck-1: use MaxPool over patches')
    # parser.add_argument('--ab_lastpool', type=str2bool, default=False,  help='Bottleneck-1: use Last-patch pooling')
    # parser.add_argument('--ab_perpatch', type=str2bool, default=False, help='No-Pool: per-patch graph (A_n) and per-patch propagation')




    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--train_only', type=bool, required=True, default=False, help='perform training on full input dataset without validation and testing')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='MS-FLOW',
                        help='model name, options: [MS-FLOW]')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT-small/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh2.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=21, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=21, help='encoder input size')
    parser.add_argument('--c_out', type=int, default=21, help='encoder input size')

    # Patching
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    parser.add_argument('--d_model', type=int, default=256, help='d_model')
    parser.add_argument('--e_layers', type=int, default=3, help='e_layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--top_k', type=int, default=10, help='top_k')
    # Moving Average
    parser.add_argument('--ma_type', type=str, default='ema', help='reg, ema, dema')
    parser.add_argument('--alpha', type=float, default=0.3, help='alpha')
    parser.add_argument('--beta', type=float, default=0.3, help='beta')


    # TimeMosaic
    # parser.add_argument('--fc_dropout', type=float, default=0.1, help='fc_dropout')
    # parser.add_argument('--fixed_weight', type=bool, default=False, help='fixed task emb weight')
    # parser.add_argument('--adjust_lr', action='store_true', default=True, help='adjust learnring rate')
    # parser.add_argument('--num_latent_token', type=int, default=4, help='Number of prompt tokens')
    # parser.add_argument('--scale_rate', type=float, default=0.001, help='emb init scale rate')
    # parser.add_argument('--patch_len_list', type=str, default='[8,16,32]',
    #                 help='List of candidate patch lengths for adaptive splitting')
    # parser.add_argument('--mask_ratio', type=float, default=0, help='mask_ratio')
    # parser.add_argument('--mask_ratio_patch', type=float, default=0, help='mask_ratio_patch')
    # parser.add_argument('--pre96', type=int, default=32, help='')
    # parser.add_argument('--pre192', type=int, default=64, help='')
    # parser.add_argument('--pre336', type=int, default=168, help='')
    # parser.add_argument('--pre720', type=int, default=240, help='')
    # parser.add_argument('--pre12', type=int, default=6, help='')
    # parser.add_argument('--counts', type=int, default=0, help='')
    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--warmup_epochs',type=int,default = 10)

    parser.add_argument('--seasonal_patterns', type=str, default='Daily', help='subset for M4')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)
    if args.data == 'm4':
        Exp = Exp_Short_Term_Forecast
    else:
        Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.des, ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}'.format(args.model_id,
                                                            args.model,
                                                            args.data,
                                                            args.features,
                                                            args.seq_len,
                                                            args.label_len,
                                                            args.pred_len,
                                                            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()