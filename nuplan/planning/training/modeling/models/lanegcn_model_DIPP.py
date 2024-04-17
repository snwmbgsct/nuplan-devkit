from typing import List, Optional, cast

import torch
from torch import nn

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.models.lanegcn_utils import (
    Actor2ActorAttention,
    Actor2LaneAttention,
    Lane2ActorAttention,
    LaneNet,
    LinearWithGroupNorm,
)
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.agents_feature_builder import AgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import GenericAgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneOnRouteStatusData,
    LaneSegmentTrafficLightData,
)
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)

import numpy as np
from torch import nn, optim
from nuplan.planning.training.modeling.models.utils.train_utils import *
from nuplan.planning.training.modeling.dipp_model.planner import *
from nuplan.planning.training.modeling.dipp_model.predictor import *
from torch.utils.data import DataLoader
from collections import defaultdict
    


def convert_predictions_to_trajectory(predictions: torch.Tensor) -> torch.Tensor:
    """
    Convert predictions tensor to Trajectory.data shape
    :param predictions: tensor from network
    :return: data suitable for Trajectory
    """
    num_batches = predictions.shape[0] # prediction 8*48 
    return predictions.view(num_batches, -1, Trajectory.state_size()) # 8*16*3 bs*num_poses*(x,y,heading)


class LaneGCN(TorchModuleWrapper):
    """
    Vector-based model that uses a series of MLPs to encode ego and agent signals, a lane graph to encode vector-map
    elements and a fusion network to capture lane & agent intra/inter-interactions through attention layers.
    Dynamic map elements such as traffic light status and ego route information are also encoded in the fusion network.

    Implementation of the original LaneGCN paper ("Learning Lane Graph Representations for Motion Forecasting").
    """

    def __init__(
        self,
        map_net_scales: int,
        num_res_blocks: int,
        num_attention_layers: int,
        a2a_dist_threshold: float,
        l2a_dist_threshold: float,
        num_output_features: int,
        feature_dim: int,
        vector_map_feature_radius: int,
        vector_map_connection_scales: Optional[List[int]],
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
    ):
        """
        :param map_net_scales: Number of scales to extend the predecessor and successor lane nodes.
        :param num_res_blocks: Number of residual blocks for the GCN (LaneGCN uses 4).
        :param num_attention_layers: Number of times to repeatedly apply the attention layer.
        :param a2a_dist_threshold: [m] distance threshold for aggregating actor-to-actor nodes
        :param l2a_dist_threshold: [m] distance threshold for aggregating map-to-actor nodes
        :param num_output_features: number of target features
        :param feature_dim: hidden layer dimension
        :param vector_map_feature_radius: The query radius scope relative to the current ego-pose.
        :param vector_map_connection_scales: The hops of lane neighbors to extract, default 1 hop
        :param past_trajectory_sampling: Sampling parameters for past trajectory
        :param future_trajectory_sampling: Sampling parameters for future trajectory
        """
        super().__init__(
            feature_builders=[
                VectorMapFeatureBuilder(
                    radius=vector_map_feature_radius,
                    connection_scales=vector_map_connection_scales,
                ),
                GenericAgentsFeatureBuilder(['VEHICLE'],trajectory_sampling=past_trajectory_sampling),
            ],
            target_builders=[EgoTrajectoryTargetBuilder(future_trajectory_sampling=future_trajectory_sampling)],
            future_trajectory_sampling=future_trajectory_sampling,
        )

        # LaneGCN components
        self.feature_dim = feature_dim
        self.connection_scales = (
            list(range(map_net_scales)) if vector_map_connection_scales is None else vector_map_connection_scales
        )
        # +1 on input dim for both agents and ego to include both history and current steps
        self.ego_input_dim = (past_trajectory_sampling.num_poses + 1) * GenericAgents.ego_state_dim() # (4+1)*3 = 15 
        self.agent_input_dim = (past_trajectory_sampling.num_poses + 1) * GenericAgents.agents_states_dim() # (4+1)*8 = 40

        self._future_steps = future_steps

        # agent layer
        self.vehicle_net = AgentEncoder()
        self.pedestrian_net = AgentEncoder()
        self.cyclist_net = AgentEncoder()

        # map layer
        self.lane_net = LaneEncoder()
        self.crosswalk_net = CrosswalkEncoder()
        
        # attention layers
        self.agent_map = Agent2Map()
        self.agent_agent = Agent2Agent()

        # decode layers
        self.plan = AVDecoder(self._future_steps)
        self.predict = AgentDecoder(self._future_steps)
        self.score = Score()
        self.lane_net_ = LaneNet(
            lane_input_len=2,
            lane_feature_len=self.feature_dim,
            num_scales=map_net_scales,
            num_residual_blocks=num_res_blocks,
            is_map_feat=False,
        )

 

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Predict
        :param features: input features containing
                        {
                            "vector_map": VectorMap,
                            "generic_agents": GenericAgents,
                        }
        :return: targets: predictions from network
                        {
                            "trajectory": Trajectory,
                        }
        """
        # Recover features
        vector_map_data = cast(VectorMap, features["vector_map"]) # type -> VectorMap #NOTES: map input
        ego_agent_features = cast(GenericAgents, features["generic_agents"]) #NOTES: agents input
        ego_past_trajectory = ego_agent_features.ego  # batch_size x num_frames x 7 = 8*5*7
        # Extract batches
        batch_size = ego_agent_features.batch_size # 8 

        # Extract features
        ego_features = []

        # Map and agent features have different size across batch so we use per sample feature extraction
        for sample_idx in range(batch_size): #BS=8

            sample_ego_feature = self.ego_feature_extractor(ego_past_trajectory[sample_idx].reshape(1, -1)) #128
            sample_ego_center = ego_agent_features.get_ego_agents_center_in_sample(sample_idx) #2
            ego = ego_past_trajectory[sample_idx] #HACK:ego input 5*8

            # Check for empty vector map input
            if not vector_map_data.is_valid:
                # Create a single lane node located at (0, 0)
                num_coords = 1
                coords = torch.zeros(
                    (num_coords, 2, 2),  # <num_lanes, 2, 2>
                    device=sample_ego_feature.device,
                    dtype=sample_ego_feature.dtype,
                    layout=sample_ego_feature.layout,
                )
                connections = {}
                for scale in self.connection_scales:
                    connections[scale] = torch.zeros((num_coords, 2), device=sample_ego_feature.device).long() 
                lane_meta_tl = torch.zeros(
                    (num_coords, LaneSegmentTrafficLightData._encoding_dim), device=sample_ego_feature.device
                )
                lane_meta_route = torch.zeros(
                    (num_coords, LaneOnRouteStatusData._encoding_dim), device=sample_ego_feature.device
                )
                lane_meta = torch.cat((lane_meta_tl, lane_meta_route), dim=1) 
            else:
                coords = vector_map_data.coords[sample_idx] #5723*2*2 num*start&end*xy
                connections = vector_map_data.multi_scale_connections[sample_idx] #4*5715*2
                lane_meta_tl = vector_map_data.traffic_light_data[sample_idx] # Traffic light data, torch.Size([5723, 4])
                lane_meta_route = vector_map_data.on_route_status[sample_idx] # torch.Size([5723, 2])
                lane_meta = torch.cat((lane_meta_tl, lane_meta_route), dim=1) # torch.Size([5723, 6]) #HACK: map_lanes
            # lane_features = self.lane_net(coords, connections) #8*128 
            lane_centers = coords.mean(axis=1)  #5723*2 #HACK:ref_line_info

            if ego_agent_features.has_agents('VEHICLE',sample_idx):
                sample_agents_feature = self.agent_feature_extractor( #10*128
                    ego_agent_features.get_flatten_agents_features_in_sample('VEHICLE',sample_idx) #10*40
                )
                sample_agents_center = ego_agent_features.get_agents_centers_in_sample('VEHICLE',sample_idx) #2*
            else:
                # if no agent in the sample, create a single agent with a stationary trajectory at 0s
                flattened_agents = torch.zeros(
                    (1, self.agent_input_dim),
                    device=sample_ego_feature.device,
                    dtype=sample_ego_feature.dtype,
                    layout=sample_ego_feature.layout,
                )
                sample_agents_feature = self.agent_feature_extractor(flattened_agents)
                sample_agents_center = torch.zeros_like(sample_ego_center).unsqueeze(dim=0)
            neighbors=ego_agent_features.get_nonflatten_agents_features_in_sample('VEHICLE',sample_idx) #torch.Size([10, 5, 8])  #HACK:DIPP neightbors
            
            
            ego_actor = self.vehicle_net(ego)
            vehicles = torch.stack([self.vehicle_net(neighbors[:, i]) for i in range(10)], dim=1) 
            pedestrians = torch.stack([self.pedestrian_net(neighbors[:, i]) for i in range(10)], dim=1) 
            cyclists = torch.stack([self.cyclist_net(neighbors[:, i]) for i in range(10)], dim=1)
            neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==2, pedestrians, vehicles)
            neighbor_actors = torch.where(neighbors[:, :, -1, -1].unsqueeze(2)==3, cyclists, neighbor_actors)
            actors = torch.cat([ego_actor.unsqueeze(1), neighbor_actors], dim=1)
            actor_mask = torch.eq(torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1), 0)[:, :, -1, -1]
            
            # maps
            lane_feature = self.lane_net_(coords, connections) #8*128 
            crosswalk_feature = self.crosswalk_net(map_crosswalks)
            lane_mask = torch.eq(map_lanes, 0)[:, :, :, 0, 0]
            crosswalk_mask = torch.eq(map_crosswalks, 0)[:, :, :, 0, 0]
            map_mask = torch.cat([lane_mask, crosswalk_mask], dim=2)
            map_mask[:, :, 0] = False # prevent nan
            
            
            
            
            
            
            ego_agents_feature = torch.cat([sample_ego_feature, sample_agents_feature], dim=0) #NOTES:input #11*128
            ego_agents_center = torch.cat([sample_ego_center.unsqueeze(dim=0), sample_agents_center], dim=0) #NOTES:input #11*2

            lane_features = self.actor2lane_attention(
                ego_agents_feature, ego_agents_center, lane_features, lane_meta, lane_centers
            )
            ego_agents_feature = self.lane2actor_attention(
                lane_features, lane_centers, ego_agents_feature, ego_agents_center
            )
            ego_agents_feature = self.actor2actor_attention(ego_agents_feature, ego_agents_center)
            ego_features.append(ego_agents_feature[0]) #8*128

        ego_features = torch.cat(ego_features).view(batch_size, -1)

        # Regress final future trajectory
        predictions = self._mlp(ego_features)
        predictor = Predictor(50) #.to('GPU')
        plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks)
        plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1) #bicycle_model(control, current_state)

        return {"trajectory": Trajectory(data=convert_predictions_to_trajectory(predictions))}
