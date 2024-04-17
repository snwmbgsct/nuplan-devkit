import numpy as np
from typing import List, Type, Optional, Tuple
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects
from nuplan.planning.simulation.trajectory.predicted_trajectory import PredictedTrajectory


class SimplePredictor(AbstractPredictor):
    def __init__(self, ego_state: EgoState, observations: Observation, duration: float, sample_time: float) -> None:
        self._ego_state = ego_state
        self._observations = observations
        self._duration = duration
        self._sample_time = sample_time
        self._occupancy_map_radius = 40
        
    def predict(self):
        """Inherited, see superclass."""
        if isinstance(self._observations, DetectionsTracks):
            objects_init = self._observations.tracked_objects.tracked_objects
            objects = [
                object
                for object in objects_init
                if np.linalg.norm(self._ego_state.center.array - object.center.array) < self._occupancy_map_radius
            ]

            # TODO：1.Predicted the Trajectory of object
            for object in objects:
                predicted_trajectories = []  # predicted_trajectories : List[PredictedTrajectory]
                predicted_trajectories = predict_constant_velocity_trajectory(object, self._sample_time)  # 使用修改后的预测函数
                object.predictions = predicted_trajectories
               

            return objects

        else:
            raise ValueError(
                f"SimplePredictor only supports DetectionsTracks. Got {self._observations.detection_type()}")

        
def predict_constant_velocity_trajectory(object, sample_time):
    # 假设每个跟踪对象都有center属性和假设的velocity属性
    initial_position = np.array([object.center.x, object.center.y])
    velocity = np.array([object.velocity.x, object.velocity.y])  # 需要有速度属性或者替代逻辑
    predicted_trajectory = [PredictedTrajectory(probability=1.0, waypoints=initial_position + velocity * sample_time)]
    
    # timestamps = np.arange(0, duration + sample_time, sample_time)
    # positions = [initial_position + velocity * t for t in timestamps]
    # predicted_trajectory = [PredictedTrajectory(probability=1.0, waypoints=initial_position + velocity * t) for t in timestamps]
    # predicted_trajectory = PredictedTrajectory(probability=1.0, waypoints=positions)
    
    return predicted_trajectory    