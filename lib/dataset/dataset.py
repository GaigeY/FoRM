#!/usr/bin/venv python
# -*- coding: utf-8 -*-
#
import os
import copy
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
import random
from loguru import logger

import sys

current_file_path = os.path.abspath(__file__)
PATH_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
if PATH_PROJECT_ROOT not in sys.path:
    sys.path.append(PATH_PROJECT_ROOT)
from lib.math.angular import rotation_matrix_to_r6d


class DatasetWindow4OneProject(Dataset):
    def __init__(self, dataset_dir, windows_length, split_ratio=(0.8, 0.1, 0.1),
                 split_strategy='merge', valid_config=None, test_config=None,
                 used_param=('vid', 'poses', 'forces_4x2'),
                 splt_param=('poses', 'forces_4x2'),
                 out_param=('poses', 'forces_4x2'),
                 window_test=True):
        self.windows_length = windows_length
        self.split_ratio = split_ratio
        self.split_strategy = split_strategy
        self.window_test = window_test

        self.train_data, self.valid_data, self.test_data = None, None, None
        self.__dataset_type = "train"
        self.used_param = used_param
        self.splt_param = splt_param
        self.out_param = out_param

        self.files = self._get_files(dataset_dir)
        # self.data:
        # {'S1': {'action1': {'poses': [tensor_p1, tensor_p2, ...], 'forces': [tensor_f1, tensor_f2, ...]},
        #         'action2': {...},
        #         ...}
        #  'S2': {...},
        #  ...}
        self.data = self._load_data()
        self._split_n_window_data(valid_config, test_config)  # If test set is not set, self.test_data = None.

    @staticmethod
    def _get_files(dataset_dir):
        # 所有后缀为`.pth`的文件的地址
        files = []
        for root, dirs, filenames in os.walk(dataset_dir):
            files.extend([os.path.join(root, filename) for filename in filenames if filename.endswith('.pth')])
        return files

    def _load_data(self):
        data_dict = {key: [] for key in self.used_param}
        for file in self.files:
            # Example: file = '/path/to/dataset/subject_name/.../example.pth'
            file_parts = file.split(os.path.sep)  # Use os.path.sep to adapt different systems.

            # Extract subject name and file name
            subject_name = next((_dir for _dir in file_parts if _dir.startswith('S')), None)  # Subject#num / S#num
            file_name = next((_dir.split('.')[0] for _dir in file_parts if _dir.endswith('.pth')), None)

            # Check subject name and file name
            if subject_name is None or file_name is None:
                raise NameError(f"Subject name {subject_name}, file name {file_name}. Path must be .../S*/.../*.pth!")

            # Load as torch data
            loaded_data = joblib.load(file)
            # Perform windowing on the data without overlapping.
            for key in self.used_param:
                data_dict[key].append(loaded_data[key])

        return data_dict

    def _split_n_window_data(self, valid_config=None, test_config=None):
        def get_split_indexes(item_size):
            # return plit set indexes following self.split_ratio (like [0.8, 0.1, 0.1]).
            indexes = torch.arange(item_size)
            shuffle_indexes = indexes.tolist()
            random.shuffle(shuffle_indexes)

            split_index_1 = int(item_size * self.split_ratio[0])
            split_index_2 = int(item_size * (1 - self.split_ratio[2]))

            return (shuffle_indexes[: split_index_1],
                    shuffle_indexes[split_index_1: split_index_2],
                    shuffle_indexes[split_index_2:])

        def check_config():
            if ':specify' in self.split_strategy:
                if valid_config is None:
                    raise AttributeError("Please specify action files in valid_config!")
                if test_config is None:
                    logger.warning("test_config is None, no test data is available!")

            elif ':random' in self.split_strategy:
                if self.split_ratio[1] == 0:
                    raise AttributeError("Ratio of valid set is 0!")
                if self.split_ratio[2] == 0:
                    logger.warning("Ratio of test set is 0!")

        def update_data_from_list_indexes(origin_list, indexes):
            return [origin_list[x] for x in indexes]

        def update_data_from_action_indexes(target_dict, idxs):
            for __idx in idxs:
                __windowed_data = self._window_data(self.windows_length,
                                                    **{key: self.data[key][__idx] for key in self.splt_param})
                for __key in self.out_param:
                    if __key in self.splt_param:
                        target_dict[__key].extend(__windowed_data[__key])
                    else:
                        target_dict[__key].extend([self.data[__key][__idx]] * len(__windowed_data[example_key]))

            return target_dict
        
        def update_data_from_action_indexes_without_window(target_dict, idxs):
            for __idx in idxs:
                for __key in self.out_param:
                    target_dict[__key].append(self.data[__key][__idx])
            return target_dict

        def update_batch_from_action_indexes(target_dict, idxs):
            for __idx in idxs:
                __windowed_data = self._window_data(self.windows_length,
                                                    **{key: self.data[key][__idx] for key in self.splt_param})
                for __key in self.out_param:
                    if __key in self.splt_param:
                        target_dict[__key].append(__windowed_data[__key])
                    else:
                        target_dict[__key].append([self.data[__key][__idx]] * len(__windowed_data[example_key]))

            return target_dict
        
        def update_batch_from_action_indexes_without_window(target_dict, idxs):
            for __idx in idxs:
                for __key in self.out_param:
                    target_dict[__key].append(self.data[__key][__idx])
            return target_dict

        subjects = list(set([x.split('_')[0] for x in self.data['vid']]))

        _base_data = {x: [] for x in self.out_param}
        train_data = copy.deepcopy(_base_data)
        valid_data = copy.deepcopy(_base_data)
        test_data = copy.deepcopy(_base_data)
        example_key = self.splt_param[0]

        check_config()

        if self.split_strategy == 'merge:random':
            # Shuffle all data
            total_data = copy.deepcopy(_base_data)
            for idx, vid in enumerate(self.data['vid']):
                windowed_data = self._window_data(self.windows_length,
                                                  **{key: self.data[key][idx] for key in self.splt_param})
                for key in self.splt_param:
                    total_data[key].extend(windowed_data[key])
                for key in [x for x in self.out_param if x not in self.splt_param]:  # 单值
                    total_data[key].extend([self.data[key][idx]] * len(windowed_data[example_key]))
            indexes_train, indexes_valid, indexes_test = get_split_indexes(len(total_data[example_key]))

            for key in self.out_param:
                train_data[key] = update_data_from_list_indexes(total_data[key], indexes_train)
                valid_data[key] = update_data_from_list_indexes(total_data[key], indexes_valid)
                test_data[key] = update_data_from_list_indexes(total_data[key], indexes_test)

        elif self.split_strategy == 'per_file:random':
            # Shuffle inter each file
            for idx, vid in enumerate(self.data['vid']):
                windowed_data = self._window_data(self.windows_length,
                                                  **{key: self.data[key][idx] for key in self.splt_param})

                # length of poses should be same as forces
                indexes_train, indexes_valid, indexes_test = get_split_indexes(len(windowed_data[example_key]))
                for key in self.out_param:
                    if key in self.splt_param:
                        train_data[key].extend(update_data_from_list_indexes(windowed_data[key], indexes_train))
                        valid_data[key].extend(update_data_from_list_indexes(windowed_data[key], indexes_valid))
                        test_data[key].extend(update_data_from_list_indexes(windowed_data[key], indexes_test))
                    else:
                        train_data[key].extend([self.data[key][idx]] * len(indexes_train))
                        valid_data[key].extend([self.data[key][idx]] * len(indexes_valid))
                        test_data[key].extend([self.data[key][idx]] * len(indexes_test))

        elif self.split_strategy == 'per_subject:random':
            # Shuffle actions
            subject_idxs = {x: [] for x in subjects}
            for idx, vid in enumerate(self.data['vid']):
                subject_idxs[vid.split('_')[0]].append(idx)

            for subject in subjects:
                indexes_train, indexes_valid, indexes_test = get_split_indexes(len(subject_idxs[subject]))
                train_data = update_data_from_action_indexes(train_data, indexes_train)
                valid_data = update_data_from_action_indexes(valid_data, indexes_valid)
                if self.window_test:
                    test_data = update_data_from_action_indexes(test_data, indexes_test)
                else:
                    test_data = update_data_from_action_indexes_without_window(test_data, indexes_test)

        elif self.split_strategy == 'specify_file:specify':
            indexes_valid = [idx for idx, vid in enumerate(self.data['vid'])
                             if valid_config is not None and vid in valid_config]
            indexes_test = [idx for idx, vid in enumerate(self.data['vid'])
                            if test_config is not None and vid in test_config]
            indexes_train = [idx for idx, vid in enumerate(self.data['vid'])
                             if idx not in indexes_valid and idx not in indexes_test]

            train_data = update_data_from_action_indexes(train_data, indexes_train)
            valid_data = update_data_from_action_indexes(valid_data, indexes_valid)
            if self.window_test:
                test_data = update_data_from_action_indexes(test_data, indexes_test)
            else:
                test_data = update_data_from_action_indexes_without_window(test_data, indexes_test)

        elif self.split_strategy == 'specify_subject:specify':
            indexes_valid = [idx for idx, vid in enumerate(self.data['vid'])
                             if valid_config is not None and vid.split('_')[0] in valid_config]
            indexes_test = [idx for idx, vid in enumerate(self.data['vid'])
                            if test_config is not None and vid.split('_')[0] in test_config]
            indexes_train = [idx for idx, vid in enumerate(self.data['vid'])
                             if idx not in indexes_valid and idx not in indexes_test]

            train_data = update_data_from_action_indexes(train_data, indexes_train)
            valid_data = update_data_from_action_indexes(valid_data, indexes_valid)
            if self.window_test:
                test_data = update_data_from_action_indexes(test_data, indexes_test)
            else:
                test_data = update_data_from_action_indexes_without_window(test_data, indexes_test)

        elif self.split_strategy == 'specify_subject:batch':
            indexes_valid = [idx for idx, vid in enumerate(self.data['vid'])
                             if valid_config is not None and vid.split('_')[0] in valid_config]
            indexes_test = [idx for idx, vid in enumerate(self.data['vid'])
                            if test_config is not None and vid.split('_')[0] in test_config]
            indexes_train = [idx for idx, vid in enumerate(self.data['vid'])
                             if idx not in indexes_valid and idx not in indexes_test]

            train_data = update_batch_from_action_indexes(train_data, indexes_train)
            valid_data = update_batch_from_action_indexes(valid_data, indexes_valid)
            if self.window_test:
                test_data = update_batch_from_action_indexes(test_data, indexes_test)
            else:
                test_data = update_batch_from_action_indexes_without_window(test_data, indexes_test)

        self.train_data = train_data
        self.valid_data = valid_data
        if test_data != _base_data:
            self.test_data = test_data

    @staticmethod
    def _window_data(windows_length=None, overlap=0, **kwargs):
        # Implement windowing logic here based on self.windows_length
        # Example:
        # [Input] >>> self._window_data(40, poses=data['poses'], grf=data['grf'])
        # [Output] >>> {'poses': [[xxx, xxx, ...], ...], 'grf': [[xxx, xxx, ...], ...]}
        # [Input] >>> self._window_data(None, poses=[4, 2, 4])
        # [Output] >>> {'poses': [[4, 2, 4]]}
        if windows_length is None:
            return {key: [value, ] for key, value in kwargs.items()}

        total_length = min(value.shape[0] for key, value in kwargs.items())

        real_window = (1 - overlap) * windows_length
        start_frames = torch.arange(int((total_length - overlap * windows_length) / real_window)) * real_window

        return {key: [value[x: x + windows_length] for x in start_frames] for key, value in kwargs.items()}

    def choose_data_type(self, dataset_type='train'):
        if dataset_type not in ['train', 'valid', 'test']:
            raise ValueError('Unsupported dataset_type, must be "train", "valid" or "test".')
        self.__dataset_type = dataset_type

    def __len__(self):
        length = {'train': len(self.train_data[self.out_param[0]]),
                  'valid': len(self.valid_data[self.out_param[0]]),
                  'test': len(self.test_data[self.out_param[0]]) if self.test_data is not None else 0}
        return length[self.__dataset_type]

    def __getitem__(self, index):
        __out_data = {'train': self.train_data,
                      'valid': self.valid_data,
                      'test': self.test_data}
        return {key: __out_data[self.__dataset_type][key][index] for key in self.out_param}


class DatasetWindowFoRM(DatasetWindow4OneProject):
    def __init__(self, dataset_dir, windows_length=None, split_ratio=(0.8, 0.1, 0.1),
                 split_strategy='merge:random', valid_config=None, test_config=None,
                 used_param=('forces_raw', 'forces_4x2', 'forces_grf', 'poses'),
                 splt_param=('forces_raw', 'forces_4x2', 'forces_grf', 'poses'),
                 out_param=('forces_raw', 'forces_4x2', 'forces_grf', 'poses'),
                 window_test=False):
        super().__init__(dataset_dir, windows_length, split_ratio, split_strategy, valid_config, test_config,
                         used_param, splt_param, out_param, window_test)

        self._convert_data_ori()

    def _convert_data_ori(self):
        """
        For all objects with "ori" in self.train_data, self.valid_data, and self.test_data:
        1. Initialize the initial state to 0
        2. Convert rotation matrix to r6d representation
        """

        def __initialize(rot_tensors):
            window_size, sensor_num, _, _ = rot_tensors.shape

            first_frame = rot_tensors[0]
            first_frame_inv = torch.inverse(first_frame).unsqueeze(0)
            first_frame_inv = first_frame_inv.repeat(window_size, 1, 1, 1)

            initialized_rot = torch.matmul(first_frame_inv.view(-1, 3, 3), rot_tensors.view(-1, 3, 3))
            return rotation_matrix_to_r6d(initialized_rot).view(window_size, sensor_num, 6)

        for _data in [self.train_data, self.valid_data, self.test_data]:
            if _data is None:
                continue
            for key in _data:
                if 'ori' in key:
                    _data[key] = [__initialize(_) for _ in _data[key]]


if __name__ == '__main__':
    # Example usage
    from lib.config import config
    path_dataset = os.path.join(config.PATH_IMPLY_DATASET, 'HAMPI')
    split_params = ('forces_raw', 'forces_4x2',
                    'imu_raw_acc', 'imu_raw_gyro',
                    'gt_poses', 'gt_trans')
    return_params = split_params + ('gt_betas',)
    
    _v2_valid_actions = ['S01_FreeMove', 'S01_FrontArmRaise', 'S02_CrissCross', 'S02_LateralArmRaise',
                         'S03_BackStep', 'S03_ChestElbowTwist', 'S04_StraightStep', 'S04_LateralArmTwist',
                         'S05_SquareStep', 'S05_LeftKneeLift', 'S06_SideStep', 'S06_RightKneeLift',
                         'S07_Running', 'S07_LeftLunge', 'S08_ObstacleDodge', 'S08_RightLunge',
                         'S09_JumpJacks', 'S09_SquatStand', 'S10_CrissCross', 'S10_Boxing',
                         'S11_FreeMove', 'S11_ChairSitStand', 'S12_BackStep', 'S12_DoorOpening',
                         'S13_SquareStep', 'S13_LeftKneeLift', 'S14_SideStep', 'S14_RightKneeLift',
                         'S15_Running', 'S15_LeftLunge', 'S16_ObstacleDodge', 'S16_RightLunge',
                         'S17_JumpJacks', 'S17_SquatStand', 'S18_CrissCross', 'S18_Boxing',
                         ]
    _v2_test_actions = ['S11_FreeMoveOutdoor']
    
    dataset = DatasetWindowFoRM(
        dataset_dir=path_dataset, windows_length=120,
        # random split
        # split_strategy='merge:random', split_ratio=(0.8, 0.1, 0.1), 
        # subject split
        # split_strategy='specify_subject:specify', valid_config=['S01', ], test_config=['S02', ],
        # use unseen subject for valid and use `freemove` for test
        split_strategy='specify_file:specify',
        valid_config=_v2_valid_actions, 
        test_config=_v2_test_actions,
        used_param=('forces_raw', 'forces_4x2',
                    'imu_raw_acc', 'imu_raw_gyro',
                    'gt_poses', 'gt_betas', 'gt_trans', 'gt_gender',
                    'framerate', 'vid'),
        splt_param=split_params,
        out_param=return_params,
        window_test=True,
    )

    # print("Train")
    # dataset.choose_data_type("train")
    # train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    # for i, data in enumerate(train_loader):
    #     print(i, [(_, x.size()) for _, x in data.items()])
    
    print("Valid")
    dataset.choose_data_type("valid")
    valid_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    # valid_loader = DataLoader(dataset)
    for i, data in enumerate(valid_loader):
        print('******** ', i, ' ********')
        if i < 1:
            for _, x in data.items():
                print(_, x.shape)
        else:
            print('forces_raw', data['forces_raw'].shape)
