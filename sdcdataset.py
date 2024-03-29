# -*- coding: utf-8 -*-


import os

from typing import Mapping, Union, Text, Any, \
Iterable, Tuple, Callable, Optional, Sequence, List




import numpy as np
import math

Shape = Sequence[int]
Shape = Union[int, Shape]
Scalar = Union[float, int]


from collections import defaultdict
from functools import partial
import json
import torch

from sdc.cache_metadata import MetadataCache
from sdc.dataset import load_datasets, load_dataloaders

from sdc.dataset import torch_cast_to_dtype
from sdc.oatomobile.torch import transforms


from sdc.constants import SCENE_TAG_TYPE_TO_OPTIONS, VALID_TRAJECTORY_TAGS
from ysdc_dataset_api.features import FeatureProducerBase
from ysdc_dataset_api.proto import get_tags_from_request, proto_to_dict, \
PedestrianTrack, Scene, VehicleTrack
from ysdc_dataset_api.utils import (
    get_file_paths,
    get_gt_trajectory,
    get_latest_track_state_by_id,
    get_to_track_frame_transform,
    read_feature_map_from_file,
    request_is_valid,
    scenes_generator,
    transform_2d_points,
)

def get_past_data_t(scene: Scene, track_id: int) -> np.ndarray:
    """Extracts a ground truth trajectory from scene for an object with track_id.
    Args:
        scene (Scene): scene to extract trajectory from
        track_id (int): track id to extract trajectory for
    Returns:
        np.ndarray: array of shape (N, 2), where N is specified by dataset
    """
    horizon = len(scene.past_vehicle_tracks)

    data = torch.zeros(2*horizon, 8)

    for t in range(horizon):
        for track in scene.past_vehicle_tracks[t].tracks:
            if track.track_id == track_id:
                data[t][0] = track.position.x
                data[t][1] = track.position.y
                data[t][2] = track.linear_velocity.x
                data[t][3] = track.linear_velocity.y
                data[t][4] = track.linear_acceleration.x
                data[t][5] = track.linear_acceleration.y
                data[t][6] = track.yaw
                data[t][7] = 1.
    return data

class SceneDataset(torch.utils.data.IterableDataset):
    def __init__(
            self,
            dataset_path: str,
            scene_tags_fpath: str,
            feature_producer: FeatureProducerBase = None,
            prerendered_dataset_path: str = None,
            transform_ground_truth_to_agent_frame: bool = True,
            scene_tags_filter: Union[Callable, None] = None,
            trajectory_tags_filter: Union[Callable, None] = None,
            pre_filtered_scene_file_paths: Optional[List[str]] = None,
            yield_metadata=False,
            A = 100
    ):
        """Pytorch-style dataset class for the motion prediction task.
        Dataset iterator performs iteration over scenes in the dataset and individual prediction
        requests in each scene. Iterator yields dict that can have the following structure:
        {
            'scene_id': str,
            'track_id': int,
            'scene_tags': Dict[str, str],
            'ground_truth_trajectory': np.ndarray,
            'prerendered_feature_map': np.ndarray,
            'feature_maps': np.ndarray,
        }.
        'scene_id' unique scene identifier.
        'track_id' vehicle id of the current prediction request.
        'ground_truth_trajectory' field is always included, it contains ground truth trajectory for
        the current prediction request.
        'prerendered_feature_map' field would be present if prerendered_dataset_path was specified,
        contains pre-rendered feature maps.
        'feature_maps' field would be present if user passes an instance of
        ysdc_dataset_api.features.FeatureRenderer, contains feature maps rendered on the fly by
        specified renderer instance.
        Args:
            dataset_path: path to the dataset directory
            scene_tags_fpath: path to the tags file
            feature_producer: instance of the FeatureProducerBase class,
                used to generate features for a data item. Defaults to None.
            prerendered_dataset_path: path to the pre-rendered dataset. Defaults to None.
            transform_ground_truth_to_agent_frame: whether to transform ground truth
                trajectory to an agent coordinate system or return global coordinates.
                Defaults to True.
            scene_tags_filter: function to filter dataset scenes by tags. Defaults to None.
            trajectory_tags_filter: function to filter prediction requests by trajectory tags.
                Defaults to None.
        Raises:
            ValueError: if none of feature_producer or prerendered_dataset_path was specified.
        """
        super(SceneDataset, self).__init__()

        self._feature_producer = feature_producer
        self._prerendered_dataset_path = prerendered_dataset_path
        self._transform_ground_truth_to_agent_frame = transform_ground_truth_to_agent_frame

        self._scene_tags_filter = _callable_or_trivial_filter(scene_tags_filter)
        self._trajectory_tags_filter = _callable_or_trivial_filter(trajectory_tags_filter)
        self._yield_metadata = yield_metadata

        self.A = A

        if pre_filtered_scene_file_paths is not None:
            print('Building MotionPredictionDataset with pre-filtered '
                  'scene file paths.')
            self._scene_file_paths = pre_filtered_scene_file_paths
        else:
            self._scene_file_paths = self._filter_paths(
                get_file_paths(dataset_path), scene_tags_fpath)

    @property
    def num_scenes(self) -> int:
        """Number of scenes in the dataset"""
        return len(self._scene_file_paths)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            file_paths = self._scene_file_paths
        else:
            file_paths = self._split_filepaths_by_worker(
                worker_info.id, worker_info.num_workers)

        def data_gen(_file_paths: List[str]):
            for scene, fpath in scenes_generator(_file_paths, yield_fpath=True):
                res = torch.zeros(self.A, 50, 8)
                gt = torch.zeros(self.A, 25, 2)
                mask = torch.zeros(self.A)
                i = 0
                for request in scene.prediction_requests:
                    
                    if not request_is_valid(scene, request):
                        continue
                    trajectory_tags = get_tags_from_request(request)
                    if not self._trajectory_tags_filter(trajectory_tags):
                        continue
                    #track = get_latest_track_state_by_id(scene, request.track_id)
                    #print(track)
                    #to_track_frame_tf = get_to_track_frame_transform(track)
                    ground_truth_trajectory = torch.tensor(get_gt_trajectory(scene, request.track_id))
                    # pos, vels, accs, yaw, mask = get_past_data(scene, request.track_id)
                    # if self._transform_ground_truth_to_agent_frame:
                    #     ground_truth_trajectory = transform_2d_points(
                    #         ground_truth_trajectory, to_track_frame_tf)
                    # result = {
                    #     'gt_trajectory': ground_truth_trajectory,
                    #     'past_trajectory' : pos,
                    #     'past_vels' : vels,
                    #     'past_accs' : accs,
                    #     'past_yaw' : yaw,
                    #     'mask' : mask
                        #'scene_id': scene.id,
                        #'track_id': request.track_id,
                        #'scene_tags': proto_to_dict(scene.scene_tags),
                    #}

                    tensor = get_past_data_t(scene, request.track_id)



                    if self._prerendered_dataset_path:
                        fm_path = self._get_serialized_fm_path(fpath, scene.id, request.track_id)
                        result['prerendered_feature_map'] = read_feature_map_from_file(fm_path)

                    if self._feature_producer:
                        result.update(
                            self._feature_producer.produce_features(scene, to_track_frame_tf))

                    if self._yield_metadata:
                        result = (
                            self.add_metadata_to_batch(
                                scene=scene, request=request,
                                trajectory_tags=trajectory_tags,
                                batch=result))
                    
                    res[i, :, :] = tensor
                    gt[i, :, :] = ground_truth_trajectory
                    mask[i] = 1.
                    i += 1
                    
                yield res, gt, mask

        return data_gen(file_paths)

    def add_metadata_to_batch(self, scene, request, trajectory_tags, batch):
        batch['scene_id'] = scene.id
        batch['request_id'] = request.track_id

        # Note that some will be "invalid"
        batch['num_vehicles'] = len(scene.prediction_requests)

        scene_tags_dict = proto_to_dict(scene.scene_tags)
        for scene_tag_type in SCENE_TAG_TYPE_TO_OPTIONS.keys():
            scene_tag_options = SCENE_TAG_TYPE_TO_OPTIONS[scene_tag_type]

            for scene_tag_option in scene_tag_options:
                try:
                    batch[f'{scene_tag_type}__{scene_tag_option}'] = int(
                        scene_tags_dict[scene_tag_type] == scene_tag_option)
                except KeyError:
                    batch[f'{scene_tag_type}__{scene_tag_option}'] = -1

        trajectory_tags = set(trajectory_tags)
        for trajectory_tag in VALID_TRAJECTORY_TAGS:
            batch[trajectory_tag] = (trajectory_tag in trajectory_tags)

        return batch

    def _get_serialized_fm_path(self, scene_fpath, scene_id, track_id):
        base, _ = os.path.split(scene_fpath)
        _, subdir = os.path.split(base)
        return os.path.join(self._prerendered_dataset_path, subdir, f'{scene_id}_{track_id}.npy')

    def _split_filepaths_by_worker(self, worker_id, num_workers):
        n_scenes_per_worker = self.num_scenes // num_workers
        split = list(range(0, self.num_scenes, n_scenes_per_worker))
        start = split[worker_id]
        if worker_id == num_workers - 1:
            stop = self.num_scenes
        else:
            stop = split[worker_id + 1]
        return self._scene_file_paths[start:stop]

    def _callable_or_lambda_true(self, f):
        if f is None:
            return lambda x: True
        if not callable(f):
            raise ValueError('Expected callable, got {}'.format(type(f)))
        return f

    def _filter_paths(self, file_paths, scene_tags_fpath):
        valid_indices = []
        with open(scene_tags_fpath, 'r') as f:
            for i, line in enumerate(f):
                tags = json.loads(line.strip())
                if self._scene_tags_filter(tags):
                    valid_indices.append(i)

        print(
            f'{len(valid_indices)}/{len(file_paths)} '
            f'scenes fit the filter criteria.')
        return [file_paths[i] for i in valid_indices]


def _callable_or_trivial_filter(f):
    if f is None:
        return _trivial_filter
    if not callable(f):
        raise ValueError('Expected callable, got {}'.format(type(f)))
    return f


def _trivial_filter(x):
    return True

