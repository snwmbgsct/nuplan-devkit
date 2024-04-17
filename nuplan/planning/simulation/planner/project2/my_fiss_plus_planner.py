import math
import time
import logging
from typing import List, Type, Optional, Tuple
from queue import PriorityQueue

import numpy as np
import numpy.typing as npt
import threading

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.scenario_builder.test.mock_abstract_scenario import MockAbstractScenario
from nuplan.common.actor_state.state_representation import StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.simulation.planner.project2.data_utils import *
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

from nuplan.planning.simulation.planner.fiss_plus.planners.fiss_plus_planner import FissPlusPlanner, FissPlusPlannerSettings, Vehicle#, nuPlan_Vehicle
from nuplan.planning.simulation.planner.fiss_plus.planners.frenet_optimal_planner import FrenetOptimalPlanner
from nuplan.planning.simulation.planner.fiss_plus.planners.common.scenario.frenet import FrenetState, State, FrenetTrajectory
from nuplan.planning.simulation.planner.fiss_plus.planners.frenet_optimal_planner import Stats

logger = logging.getLogger(__name__)


class MyFissPlusPlanner(AbstractPlanner, FissPlusPlanner):
    """
    Fiss Plus Planner.
    """

    def __init__(
            self,
            scenario: AbstractScenario,
            horizon_seconds: float,
            sampling_time: float,
            max_velocity: float = 5.0,
            vehicle: Vehicle = Vehicle(get_pacifica_parameters()),
            planner_settings: FissPlusPlannerSettings = FissPlusPlannerSettings()       
        
            
    ):
        super(MyFissPlusPlanner, self).__init__(planner_settings, vehicle) # ?：如何使用这些settings
        # FissPlusPlanner.__init__(self, planner_settings, vehicle)# 
        """
        Constructor for SimplePlanner.
        :param horizon_seconds: [s] time horizon being run.
        :param sampling_time: [s] sampling timestep.
        :param max_velocity: [m/s] ego max velocity.
        """
        # self._lock = threading.Lock()
        self._scenario = scenario
        self.horizon_time = TimePoint(int(horizon_seconds * 1e6)) #TimePoint(time_us=8000000)
        self.sampling_time = TimePoint(int(sampling_time * 1e6))  #TimePoint(time_us=100000)
        self.num_samples = int(self.horizon_time.time_us/self.sampling_time.time_us) # 80
        self.max_velocity = max_velocity
        self.vehicle = Vehicle(get_pacifica_parameters())

        self._router: Optional[BFSRouter] = None
        self._predictor: AbstractPredictor = None
        self._reference_path_provider: Optional[ReferenceLineProvider] = None
        self._routing_complete = False
        self.settings.tick_t=1
        self.path_gen_flag=False
        
        

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



    # TODO: 2. Please implement your own trajectory planning.
    def plan(self, frenet_state: FrenetState, max_target_speed: float, obstacles: list, time_step_now: int=0) -> FrenetTrajectory:
        
        t_start = time.time()
        
        # Reset values for each planning cycle
        self.stats = Stats() 
        self.settings.highest_speed = max_target_speed
        self.start_state = frenet_state
        self.frontier_idxs = PriorityQueue()
        self.candidate_trajs = PriorityQueue()
        self.refined_trajs = PriorityQueue()
        self.best_traj = None
        
        # Sample all the end states in 3 dimension, [d, v, t] and form the 3d traj canndidate array
        self.trajs_3d = self.sample_end_frenet_states()
        self.sizes = np.array([len(self.trajs_3d), len(self.trajs_3d[0]), len(self.trajs_3d[0][0])])
        
        best_idx = None
        best_traj_found = False
        
        while not best_traj_found:
            self.stats.num_iter += 1
            
            ############################## Initial Guess ##############################
            
            if self.candidate_trajs.empty():
                best_idx = self.find_initial_guess()
                if best_idx is None:
                    break
            else:
                best_idx = self.candidate_trajs.queue[0][1] # peek the index of the most likely candidate
                    
            i = 0
            converged = False
            while not converged:
                i += 1
                is_minimum, best_idx = self.explore_neighbors(best_idx)
                if self.frontier_idxs.empty():
                    converged = True
                else:
                    _, best_idx = self.frontier_idxs.get()
                        
            ############################## Validation ##############################
            
            if not self.candidate_trajs.empty():
                _, idx = self.candidate_trajs.get()
                candidate = self.trajs_3d[idx[0]][idx[1]][idx[2]]
                self.stats.num_trajs_validated += 1
                # Convert to global coordinates
                candidate = self.calc_global_paths([candidate])
                # Check for constraints
                passed_candidate = self.check_constraints(candidate)
                if passed_candidate:
                    # Check for collisions
                    safe_candidate = self.check_collisions(candidate, obstacles, time_step_now) 
                    self.stats.num_collison_checks += 1
                    if safe_candidate:
                        best_traj_found = True
                        self.best_traj = safe_candidate[0] ## lowest cost ranks first
                        self.prev_best_idx = self.best_traj.idx
                        break
                    else:
                        continue
                else:
                    continue
            else:
                break
            
        if best_traj_found and self.settings.refine_trajectory:
            time_spent = time.time() - t_start
            time_left = self.settings.time_limit - time_spent
            
            if not self.settings.has_time_limit or time_left > 0.0:
                refined_traj = self.refine_solution(self.best_traj, time_left, obstacles, time_step_now) 
                if refined_traj is not None:
                    self.best_traj = refined_traj
        self.frontier_idxs = None
        self.candidate_trajs = None
        self.refined_trajs = None                 
        return self.best_traj
    
    def do_planning(self, ego_state, objects: tuple, num_samples: tuple):
        
        max_speed = 13.5      
        
        ego_lane_pts = np.array([[state.x, state.y, state.heading] for idx, state in enumerate(self._reference_path_provider._discrete_path) if idx % 50 == 0])
        _, indices = np.unique(ego_lane_pts[..., :2], axis=0, return_index=True)
        ego_lane_pts = ego_lane_pts[np.sort(indices)]
        if self.path_gen_flag == False:
            t_now = time.time()
            _, self.ref_ego_lane_pts = self.generate_frenet_frame(ego_lane_pts)        
            t_after = time.time()
            # print("Time to generate_frenet_frame: ", t_after - t_now)
            self.path_gen_flag=True
        else:
            pass
        # Initial state
        start_state = State(t=ego_state.time_point.time_s, x=ego_state.center.x, y=ego_state.center.y, yaw=ego_state.center.heading, v=ego_state.dynamic_car_state.speed, a=ego_state.dynamic_car_state.acceleration)
        current_frenet_state = FrenetState()
        current_frenet_state.from_state(start_state, self.ref_ego_lane_pts)
        current_time_point = ego_state.time_point
        state_list = [ego_state]
        
        
        for i in range(int(self.horizon_time.time_us / self.sampling_time.time_us)):
            t_now = time.time()
            best_traj_ego = self.plan(current_frenet_state, max_speed, objects, i)
            t_after = time.time()
            print("Time to planning: ", t_after - t_now)
            next_step_idx = 1
            current_state = best_traj_ego.state_at_time_step(next_step_idx)
            current_frenet_state = best_traj_ego.frenet_state_at_time_step(
                next_step_idx)
            current_time_point += self.sampling_time 
            # state = EgoState(np.array([current_state.x, current_state.y, current_state.yaw]), DynamicCarState(self.vehicle.a, current_state.v, current_state.a), tire_steering_angle=0.0, is_in_auto_mode=True, time_point=current_time_point)
            state = EgoState.build_from_center(
            center=StateSE2(current_state.x, current_state.y, current_state.yaw),
            center_velocity_2d=StateVector2D(current_state.v, 0),
            center_acceleration_2d=StateVector2D(0, 0),
            tire_steering_angle=0.0,
            time_point=current_time_point,
            vehicle_parameters=get_pacifica_parameters(),
        )
            # current_time_point += self.sampling_time 
            state_list.append(state)
        t_after = time.time()
        # print("Time to planning: ", t_after - t_now)

        return state_list
                    
    # def get_neighbor_agents_future(self, agent_index):
    #     current_ego_state = self._scenario.initial_ego_state
    #     present_tracked_objects = self._scenario.initial_tracked_objects.tracked_objects

    #     # Get all future poses of of other agents
    #     future_tracked_objects = [
    #         tracked_objects.tracked_objects
    #         for tracked_objects in self._scenario.get_future_tracked_objects(
    #             iteration=0, time_horizon=self.future_time_horizon, num_samples=self.num_future_poses
    #         )
    #     ]

    #     sampled_future_observations = [present_tracked_objects] + future_tracked_objects
    #     future_tracked_objects_tensor_list, _ = sampled_tracked_objects_to_tensor_list(sampled_future_observations)
    #     agent_futures = agent_future_process(current_ego_state, future_tracked_objects_tensor_list, self.num_agents, agent_index)

    #     return agent_futures    
    
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
        # self._predictor = SimplePredictor(ego_state, observations, self.horizon_time.time_s, self.sampling_time.time_s)
        # objects = self._predictor.predict()
        
        # 3. Perfect prediction
        # current_state = self._scenario.get_ego_state_at_iteration(current_input.iteration.index)
        
        present_tracked_objects = self._scenario.initial_tracked_objects.tracked_objects
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self._scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.horizon_time.time_s, num_samples=self.num_samples
            )
        ]
        obstacles = [present_tracked_objects] + future_tracked_objects

        # 4. Planning
        trajectory: List[EgoState] = self.do_planning(ego_state, obstacles, (5, 5, 5))
        
        
        # current_time_point = ego_state.time_point
        # state_list = [ego_state]
        # for _ in range(int(self.horizon_time.time_us / self.sampling_time.time_us)):
        #     current_time_point += self.sampling_time 
        #     # state = EgoState(np.array([current_state.x, current_state.y, current_state.yaw]), DynamicCarState(self.vehicle.a, current_state.v, current_state.a), tire_steering_angle=0.0, is_in_auto_mode=True, time_point=current_time_point)
        #     state = EgoState.build_from_center(
        #     center=StateSE2(x=331325.28775870294, y=4691000.731798908, heading=2.5610833687903742),
        #     center_velocity_2d=StateVector2D(7.479167992097267, 0),
        #     center_acceleration_2d=StateVector2D(0, 0),
        #     tire_steering_angle=0.0,
        #     time_point=current_time_point,
        #     vehicle_parameters=get_pacifica_parameters(),
        # )
        #     # current_time_point += self.sampling_time 
        #     state_list.append(state)

        return InterpolatedTrajectory(trajectory)

    