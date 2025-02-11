"""
File: parser.py
Author: OccamRazerTeam
Email: vincent.duan95@outlook.com
Github: https://github.com/VDeamoV
Description: This file is used to return the config typed in by user
"""

import argparse


def get_cifar_parser_params():
    """
    Use to return the base config
    """
    parser = argparse.ArgumentParser(description="cifar configures")
    parser.add_argument('--data', type=str, default='../data',
                        help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.025,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=0.001,
                        help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--report_freq', type=float, default=50,
                        help='report frequency')
    parser.add_argument('--gpu', type=int, default=1,
                        help='gpu device id')
    parser.add_argument('--epochs', type=int, default=50,
                        help='num of training epochs')
    parser.add_argument('--init_channels', type=int, default=16,
                        help='num of init channels')
    parser.add_argument('--layers', type=int, default=8,
                        help='total number of layers')
    parser.add_argument('--model_path', type=str, default='saved_models',
                        help='path to save the model')
    parser.add_argument('--cutout', action='store_true', default=False,
                        help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16,
                        help='cutout length')
    parser.add_argument('--drop_path_prob', type=float, default=0.3,
                        help='drop path probability')
    parser.add_argument('--save', type=str, default='EXP',
                        help='experiment name')
    parser.add_argument('--seed', type=int, default=2,
                        help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5,
                        help='gradient clipping')
    parser.add_argument('--train_portion', type=float, default=0.5,
                        help='portion of training data')
    parser.add_argument('--unrolled', action='store_true', default=False,
                        help='use one-step unrolled validation loss')
    parser.add_argument('--arch_learning_rate', type=float, default=3e-4,
                        help='learning rate for arch encoding')
    parser.add_argument('--arch_weight_decay', type=float, default=1e-3,
                        help='weight decay for arch encoding')
    args = parser.parse_args()
    return args
