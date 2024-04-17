import torch
import sys
import csv
import time
import argparse
import logging
import os
import numpy as np
from torch import nn, optim
from utils.train_utils import *
from nuplan.planning.training.modeling.dipp_model.planner import MotionPlanner
from nuplan.planning.training.modeling.dipp_model.predictor import Predictor
from torch.utils.data import DataLoader




import os
import pathlib
from collections import defaultdict
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle, AbstractModelFeature
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories
from nuplan.planning.training.preprocessing.feature_preprocessor import FeaturePreprocessor
from nuplan.planning.training.preprocessing.feature_builders.generic_agents_feature_builder import GenericAgentsFeatureBuilder
from nuplan.planning.training.preprocessing.feature_builders.vector_map_feature_builder import VectorMapFeatureBuilder
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario





folder_path = '/tmp/tutorial_nuplan_framework/cache/2021.05.12.23.36.44_veh-35_00152_00504/high_magnitude_speed'

# 创建一个字典，将文件名映射到对应的特征类
feature_class_mapping = {
    "generic_agents.gz": GenericAgents,
    "vector_map.gz": VectorMap,
    "trajectory.gz": Trajectory
}


all_ego = []
all_agents =  defaultdict(list)

for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".gz"):
            gz_file = os.path.join(root, file)
            print(f"Loading {gz_file}")
            feature_cache = FeatureCachePickle()
            feature_file = pathlib.Path(gz_file)
            # 检查文件名是否在字典中，如果是，则加载相应的特征并打印ego属性
            if file in feature_class_mapping:
                loaded_feature = feature_cache.load_computed_feature_from_folder(feature_file, feature_class_mapping[file])
                if file == "generic_agents.gz":
                    all_ego.append(loaded_feature.ego[0])
                    for agent_name, agent in loaded_feature.agents.items():
                        all_agents[agent_name].append(agent[0])
                elif file == "vector_map.gz":
                    vector_map_data = loaded_feature #vector_map_data.append(loaded_feature)
                else:
                    trajectory = loaded_feature #trajectory.append(loaded_feature)
# 创建一个汇总了所有 generic_agents.gz 文件中数据的 GenericAgents 实例
ego_agent_features = GenericAgents(ego=all_ego, agents=all_agents)

ego = ego_agent_features.ego
neighbors = ego_agent_features.agents['VEHICLE']







for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".gz"):
            gz_file = os.path.join(root, file)
            print(f"Loading {gz_file}")
            feature_cache = FeatureCachePickle()
            feature_file = pathlib.Path(gz_file)
            # 检查文件名是否在字典中，如果是，则加载相应的特征并打印ego属性
            if file in feature_class_mapping:
                loaded_feature = feature_cache.load_computed_feature_from_folder(feature_file, feature_class_mapping[file])
                if file == "generic_agents.gz":
                    ego_agent_features = loaded_feature
                    ego_agent_features.ego.append(loaded_feature.ego) #ego_agent_features = loaded_feature 
                elif file == "vector_map.gz":
                    vector_map_data = loaded_feature #vector_map_data.append(loaded_feature)
                else:
                    trajectory = loaded_feature #trajectory.append(loaded_feature)
            else:
                print("Unsupported file:", file)
 
                


ego = ego_agent_features.ego
neighbors = ego_agent_features.agents['VEHICLE']
            # Check for empty vector map input
if not vector_map_data.is_valid:
    # Create a single lane node located at (0, 0)
    print("vector_map_data is not valid") 
else:
    coords = vector_map_data.coords #5723*2*2 num*start&end*xy
    connections = vector_map_data.multi_scale_connections #4*5715*2
    lane_meta_tl = vector_map_data.traffic_light_data # Traffic light data, torch.Size([5723, 4])
    lane_meta_route = vector_map_data.on_route_status # torch.Size([5723, 2])
    lane_meta = torch.cat((lane_meta_tl, lane_meta_route), dim=1) # torch.Size([5723, 6]) #HACK: map_lanes
lane_features = self.lane_net_(coords, connections) #8*128 
lane_centers = coords.mean(axis=1)  #5723*2 #HACK:ref_line_info




def train_epoch(data_loader, predictor, planner, optimizer, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.train()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        map_lanes = batch[2].to(args.device)
        map_crosswalks = batch[3].to(args.device)
        ref_line_info = batch[4].to(args.device)
        ground_truth = batch[5].to(args.device)
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1] #ego add 1 dim
        weights = torch.ne(ground_truth[:, 1:, :, :3], 0) #batch_size, 11 agents, states, ground truth(x, y, heading)

        # predict
        optimizer.zero_grad()
        plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks)
        plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1) #bicycle_model(control, current_state)
        loss = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights) # multi-future multi-agent loss
        
        # plan
        if use_planning:
            plan, prediction = select_future(plans, predictions, scores)

            planner_inputs = {
                "control_variables": plan.view(-1, 100), # initial control sequence
                "predictions": prediction, # prediction for surrounding vehicles 
                "ref_line_info": ref_line_info,
                "current_state": current_state
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)

            final_values, info = planner.layer.forward(planner_inputs)
            plan = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3]) 
            plan_loss += F.smooth_l1_loss(plan[:, -1], ground_truth[:, 0, -1, :3])
            loss += plan_loss + 1e-3 * plan_cost # planning loss
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        # loss backward
        loss.backward()
        nn.utils.clip_grad_norm_(predictor.parameters(), 5)
        optimizer.step()

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # show loss
        current += batch[0].shape[0]
        sys.stdout.write(f"\rTrain Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
        sys.stdout.flush()

    # show metrics
    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(f'\nplannerADE: {plannerADE:.4f}, plannerFDE: {plannerFDE:.4f}, predictorADE: {predictorADE:.4f}, predictorFDE: {predictorFDE:.4f}')
        
    return np.mean(epoch_loss), epoch_metrics

def valid_epoch(data_loader, predictor, planner, use_planning):
    epoch_loss = []
    epoch_metrics = []
    current = 0
    size = len(data_loader.dataset)
    predictor.eval()
    start_time = time.time()

    for batch in data_loader:
        # prepare data
        ego = batch[0].to(args.device)
        neighbors = batch[1].to(args.device)
        map_lanes = batch[2].to(args.device)
        map_crosswalks = batch[3].to(args.device)
        ref_line_info = batch[4].to(args.device)
        ground_truth = batch[5].to(args.device) #torch.Size([32, 11, 50, 5])
        current_state = torch.cat([ego.unsqueeze(1), neighbors[..., :-1]], dim=1)[:, :, -1]
        weights = torch.ne(ground_truth[:, 1:, :, :3], 0)

        # predict
        with torch.no_grad():
            plans, predictions, scores, cost_function_weights = predictor(ego, neighbors, map_lanes, map_crosswalks)
            plan_trajs = torch.stack([bicycle_model(plans[:, i], ego[:, -1])[:, :, :3] for i in range(3)], dim=1)
            loss = MFMA_loss(plan_trajs, predictions, scores, ground_truth, weights) # multi-future multi-agent loss

        # plan 
        if use_planning:
            plan, prediction = select_future(plans, predictions, scores)

            planner_inputs = {
                "control_variables": plan.view(-1, 100), # generate initial control sequence
                "predictions": prediction, # generate predictions for surrounding vehicles 
                "ref_line_info": ref_line_info,
                "current_state": current_state
            }

            for i in range(cost_function_weights.shape[1]):
                planner_inputs[f'cost_function_weight_{i+1}'] = cost_function_weights[:, i].unsqueeze(1)

            with torch.no_grad():
                final_values, info = planner.layer.forward(planner_inputs)

            plan = final_values["control_variables"].view(-1, 50, 2)
            plan = bicycle_model(plan, ego[:, -1])[:, :, :3]

            plan_cost = planner.objective.error_squared_norm().mean() / planner.objective.dim()
            plan_loss = F.smooth_l1_loss(plan, ground_truth[:, 0, :, :3])
            plan_loss += F.smooth_l1_loss(plan[:, -1], ground_truth[:, 0, -1, :3])
            loss += plan_loss + 1e-3 * plan_cost # planning loss
        else:
            plan, prediction = select_future(plan_trajs, predictions, scores)

        # compute metrics
        metrics = motion_metrics(plan, prediction, ground_truth, weights)
        epoch_metrics.append(metrics)
        epoch_loss.append(loss.item())

        # show progress
        current += batch[0].shape[0]
        sys.stdout.write(f"\rValid Progress: [{current:>6d}/{size:>6d}]  Loss: {np.mean(epoch_loss):>.4f}  {(time.time()-start_time)/current:>.4f}s/sample")
        sys.stdout.flush()

    epoch_metrics = np.array(epoch_metrics)
    plannerADE, plannerFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictorADE, predictorFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [plannerADE, plannerFDE, predictorADE, predictorFDE]
    logging.info(f'\nval-plannerADE: {plannerADE:.4f}, val-plannerFDE: {plannerFDE:.4f}, val-predictorADE: {predictorADE:.4f}, val-predictorFDE: {predictorFDE:.4f}')

    return np.mean(epoch_loss), epoch_metrics

def model_training():
    # Logging
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)
    initLogging(log_file=log_path+'train.log')

    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use integrated planning module: {}".format(args.use_planning))
    logging.info("Use device: {}".format(args.device))

    # set seed
    set_seed(args.seed)

    # set up predictor
    predictor = Predictor(50).to(args.device) #future steps
    
    # set up planner
    if args.use_planning:
        trajectory_len, feature_len = 50, 9
        planner = MotionPlanner(trajectory_len, feature_len, args.device)
    else:
        planner = None
    
    # set up optimizer
    optimizer = optim.Adam(predictor.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)# 4 epochs, multiplied by 0.5

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders
    train_set = DrivingData(args.train_set+'/*')
    valid_set = DrivingData(args.valid_set+'/*')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))
    
    # begin training
    for epoch in range(train_epochs):
        logging.info(f"Epoch {epoch+1}/{train_epochs}")
        
        # train 
        if planner:
            if epoch < args.pretrain_epochs:
                args.use_planning = False
            else:
                args.use_planning = True         

        train_loss, train_metrics = train_epoch(train_loader, predictor, planner, optimizer, args.use_planning)
        val_loss, val_metrics = valid_epoch(valid_loader, predictor, planner, args.use_planning)

        # save to training log
        log = {'epoch': epoch+1, 'loss': train_loss, 'lr': optimizer.param_groups[0]['lr'], 'val-loss': val_loss, 
               'train-plannerADE': train_metrics[0], 'train-plannerFDE': train_metrics[1], 
               'train-predictorADE': train_metrics[2], 'train-predictorFDE': train_metrics[3],
               'val-plannerADE': val_metrics[0], 'val-plannerFDE': val_metrics[1], 
               'val-predictorADE': val_metrics[2], 'val-predictorFDE': val_metrics[3]}

        if epoch == 0:
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())

        # reduce learning rate
        scheduler.step()

        # save model at the end of epoch
        torch.save(predictor.state_dict(), f'training_log/{args.name}/model_{epoch+1}_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "Exp1")', default="Exp1")
    parser.add_argument('--train_set', type=str, help='path to train datasets', default='/media/arc/Nuclear/Waymo_processed/Waymo_Train')
    parser.add_argument('--valid_set', type=str, help='path to validation datasets', default='/media/arc/Nuclear/Waymo_processed/Waymo_Val')
    parser.add_argument('--seed', type=int, help='fix random seed', default=42)
    parser.add_argument("--num_workers", type=int, default=8, help="number of workers used for dataloader")
    parser.add_argument('--pretrain_epochs', type=int, help='epochs of pretraining predictor', default=5)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=20)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 32)', default=32)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 2e-4)', default=2e-4)
    parser.add_argument('--use_planning', action="store_true", help='if use integrated planning module (default: False)', default=True)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()

    # Run
    model_training()