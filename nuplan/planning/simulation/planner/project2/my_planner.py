import math
import logging
from typing import List, Type, Optional, Tuple

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.project2.bfs_router import BFSRouter
from nuplan.planning.simulation.planner.project2.reference_line_provider import ReferenceLineProvider
from nuplan.planning.simulation.planner.project2.simple_predictor import SimplePredictor
from nuplan.planning.simulation.planner.project2.abstract_predictor import AbstractPredictor

from nuplan.planning.simulation.planner.project2.merge_path_speed import transform_path_planning, cal_dynamic_state, cal_pose
from nuplan.common.actor_state.ego_state import DynamicCarState, EgoState
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects import TrackedObject, TrackedObjects

logger = logging.getLogger(__name__)


class MyPlanner(AbstractPlanner):
    """
    Planner going straight.
    """

    def __init__(
            self,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,
    ):
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6))
        self.sampling_time = TimePoint(int(sampling_time * 1e6))
        self.max_velocity = max_velocity

        self._router: Optional[BFSRouter] = None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        self._router = BFSRouter(initialization.map_api)
        self._router._initialize_route_plan(initialization.route_roadblock_ids)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Implement a trajectory that goes straight.
        Inherited, see superclass.
        """

        # 1. Routing
        ego_state, observations = current_input.history.current_state
        if not self._routing_complete:
            self._router._initialize_ego_path(ego_state, self.max_velocity)
            self._routing_complete = True

        # 2. Generate reference line
        self._reference_path_provider = ReferenceLineProvider(self._router)
        self._reference_path_provider._reference_line_generate(ego_state)

        # 3. Objects prediction
        self._predictor = SimplePredictor(ego_state, observations, self.horizon_time.time_s, self.sampling_time.time_s)
        objects = self._predictor.predict()

        # 4. Planning
        trajectory: List[EgoState] = self.planning(ego_state, self._reference_path_provider, objects,
                                                    self.horizon_time, self.sampling_time, self.max_velocity)

        return InterpolatedTrajectory(trajectory)
    
    def path_planning(self, ego_state: EgoState, reference_path_provider: ReferenceLineProvider) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Path planning.
        :param ego_state: Ego state.
        :param reference_path_provider: Reference path provider.
        :return: optimal_path_l, optimal_path_dl, optimal_path_ddl, optimal_path_s
        """
                    
        def find_nearest_point_idx(ego_state: EgoState, polyline: np.ndarray):
            distances = np.hypot(polyline[:,0] - ego_state.center.x, polyline[:,1] - ego_state.center.y)
            return np.argmin(distances)

        def find_next_point_idx(ego_state: ego_state, polyline: np.ndarray): #state dim and polyline dim
            nearest_idx = find_nearest_point_idx(ego_state, polyline)
            heading = np.arctan2(polyline[nearest_idx, 1] - ego_state.center.y, polyline[nearest_idx, 0]- ego_state.center.x)
            angle = abs(ego_state.center.heading - heading)
            angle = min(2*np.pi - angle, angle)

            if angle > np.pi/2:
                next_wp_id = nearest_idx + 1
            else:
                next_wp_id = nearest_idx
                
            # if it is behind the start of the waypoint list
            if next_wp_id < 1:
                next_wp_id = 1
            # if it reaches the end of the waypoint list
            elif next_wp_id >= polyline.shape[0]:
                next_wp_id = polyline.shape[0] - 1
            
            return next_wp_id
        
        def unifyAngleRange(angle):
            """
            Unifies the angle within the range of -pi to pi.

            Parameters:
            angle (float): The input angle in radians.

            Returns:
            float: The angle within the range of -pi to pi.
            """
            new_angle = angle
            while(new_angle > np.pi):
                new_angle -= 2*np.pi
            while(new_angle < -np.pi):
                new_angle += 2*np.pi
            return new_angle
        
        discrete_path = reference_path_provider._discrete_path
        polyline = np.array([[state.x, state.y, state.heading] for state in discrete_path])
        
        # Get the previous and the next waypoint ids
        next_wp_id = find_next_point_idx(ego_state, polyline)
        prev_wp_id = max(next_wp_id - 1, 0)
        # vector n from previous waypoint to next waypoint
        n_x = polyline[next_wp_id, 0] - polyline[prev_wp_id, 0]
        n_y = polyline[next_wp_id, 1] - polyline[prev_wp_id, 1]
        # vector x from previous waypoint to current position
        x_x = ego_state.center.x - polyline[prev_wp_id, 0]
        x_y = ego_state.center.y - polyline[prev_wp_id, 1]
        x_yaw = np.arctan2(x_y, x_x)
        # find the projection of x on n        
        proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
        proj_x = proj_norm * n_x
        proj_y = proj_norm * n_y

        # calculate d value
        self.d = np.hypot(x_x - proj_x, x_y - proj_y)

        wp_yaw = polyline[prev_wp_id, 2]
        delta_yaw = unifyAngleRange(ego_state.center.heading - wp_yaw)

        # if wp_yaw > x_yaw: 
        if wp_yaw <= x_yaw: # CommonRoad
            self.d *= -1

        # calculate s value
        self.s = 0
        for i in range(prev_wp_id):
            self.s += np.hypot(polyline[i+1, 0] - polyline[i, 0], polyline[i+1, 1] - polyline[i, 1])

        # calculate s_d and d_d
        self.t = ego_state.t
        self.s_d = ego_state.dynamic_car_state.speed * np.cos(delta_yaw)
        self.s_dd = 0.0
        self.s_ddd = 0.0
        self.d_d = ego_state.dynamic_car_state.speed * np.sin(delta_yaw)
        self.d_dd = 0.0
        self.d_ddd = 0.0
        
        return self.d, self.d_d, self.d_dd, self.s

        
    
    def speed_planning(self, ego_state: EgoState, horizon_time: float, max_velocity: float, objects: List[TrackedObjects], path_idx2s: npt.ArrayLike, path_x: npt.ArrayLike, path_y: npt.ArrayLike, path_heading: npt.ArrayLike, path_kappa: npt.ArrayLike) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Speed planning.
        :param ego_state: Ego state.
        :param horizon_time: Horizon time.
        :param max_velocity: Max velocity.
        :param objects: Objects.
        :param path_idx2s: Path idx to s.
        :param path_x: Path x.
        :param path_y: Path y.
        :param path_heading: Path heading.
        :param path_kappa: Path kappa.
        :return: optimal_speed_s, optimal_speed_s_dot, optimal_speed_s_2dot, optimal_speed_t
        """
        return np.array([]), np.array([]), np.array([]), np.array([])
        
        

    # TODO: 2. Please implement your own trajectory planning.
    def planning(self,
                 ego_state: EgoState,
                 reference_path_provider: ReferenceLineProvider,
                 object: List[TrackedObjects],
                 horizon_time: TimePoint,
                 sampling_time: TimePoint,
                 max_velocity: float) -> List[EgoState]:
        """
        Implement trajectory planning based on input and output, recommend using lattice planner or piecewise jerk planner.
        param: ego_state Initial state of the ego vehicle
        param: reference_path_provider Information about the reference path 
        param: objects Information about dynamic obstacles
        param: horizon_time Total planning time
        param: sampling_time Planning sampling time
        param: max_velocity Planning speed limit (adjustable according to road speed limits during planning process)
        return: trajectory Planning result
        """
        
        

        """
        """
        # 可以实现基于采样的planer或者横纵向解耦的planner，此处给出planner的示例，仅提供实现思路供参考
        # 1.Path planning
        optimal_path_l, optimal_path_dl, optimal_path_ddl, optimal_path_s = self.path_planning( \
            ego_state, reference_path_provider)

        # 2.Transform path planning result to cartesian frame
        path_idx2s, path_x, path_y, path_heading, path_kappa = transform_path_planning(optimal_path_s, optimal_path_l, \
                                                                                       optimal_path_dl,
                                                                                       optimal_path_ddl, \
                                                                                       reference_path_provider)

        # 3.Speed planning
        optimal_speed_s, optimal_speed_s_dot, optimal_speed_s_2dot, optimal_speed_t = self.speed_planning( \
            ego_state, horizon_time.time_s, max_velocity, object, \
            path_idx2s, path_x, path_y, path_heading, path_kappa)

        # 4.Produce ego trajectory
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                ego_state.dynamic_car_state.rear_axle_acceleration_2d,
            ),
            tire_steering_angle=ego_state.dynamic_car_state.tire_steering_rate,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        trajectory: List[EgoState] = [state]
        for iter in range(int(horizon_time.time_us / sampling_time.time_us)):
            relative_time = (iter + 1) * sampling_time.time_s
            # 根据relative_time 和 speed planning 计算 velocity accelerate （三次多项式）
            s, velocity, accelerate = cal_dynamic_state(relative_time, optimal_speed_t, optimal_speed_s,
                                                        optimal_speed_s_dot, optimal_speed_s_2dot)
            # 根据当前时间下的s 和 路径规划结果 计算 x y heading kappa （线形插值）
            x, y, heading, _ = cal_pose(s, path_idx2s, path_x, path_y, path_heading, path_kappa)

            state = EgoState.build_from_rear_axle(
                rear_axle_pose=StateSE2(x, y, heading),
                rear_axle_velocity_2d=StateVector2D(velocity, 0),
                rear_axle_acceleration_2d=StateVector2D(accelerate, 0),
                tire_steering_angle=heading,
                time_point=state.time_point + sampling_time,
                vehicle_parameters=state.car_footprint.vehicle_parameters,
                is_in_auto_mode=True,
                angular_vel=0,
                angular_accel=0,
            )

            trajectory.append(state)
        
        # trajectory = []
        return trajectory