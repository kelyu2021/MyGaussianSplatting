#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os

def mkdir_p(folder_path):
    try:
        makedirs(folder_path)
    except OSError as exc:
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxCheckpoint(folder):
    """Find the latest checkpoint number, supporting both epoch_ and iteration_ prefixes.

    Returns (prefix, number) where prefix is 'epoch' or 'iteration'.
    """
    best_prefix, best_num = None, -1
    for fname in os.listdir(folder):
        name = fname.split('.')[0]  # strip extension
        for prefix in ('epoch', 'iteration'):
            if name.startswith(prefix + '_'):
                try:
                    num = int(name[len(prefix) + 1:])
                    if num > best_num or (num == best_num and prefix == 'epoch'):
                        best_prefix, best_num = prefix, num
                except ValueError:
                    pass
    if best_prefix is None:
        # Legacy fallback
        saved = [int(fname.split('.')[0].split("_")[-1]) for fname in os.listdir(folder)]
        return 'iteration', max(saved)
    return best_prefix, best_num

searchForMaxIteration = searchForMaxCheckpoint  # backward compat alias
