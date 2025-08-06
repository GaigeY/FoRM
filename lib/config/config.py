#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#

import os

current_file_path = os.path.abspath(__file__)
PATH_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))

PATH_MODEL_SMPL = os.path.join(PATH_PROJECT_ROOT, 'assets', 'smplx_models')
PATH_IMPLY_DATASET = os.path.join(PATH_PROJECT_ROOT, 'data')
PATH_CHECKPOINTS = os.path.join(PATH_PROJECT_ROOT, 'checkpoint')
