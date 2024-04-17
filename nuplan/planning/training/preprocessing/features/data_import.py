import os
import pathlib
from nuplan.planning.training.preprocessing.utils.feature_cache import FeatureCachePickle, AbstractModelFeature
from nuplan.planning.training.preprocessing.features.generic_agents import GenericAgents
from nuplan.planning.training.preprocessing.features.vector_map import VectorMap
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.features.trajectories import Trajectories

folder_path = '/tmp/tutorial_nuplan_framework/cache/2021.05.12.22.00.38_veh-35_01008_01518/high_magnitude_speed/0d252d7f1b0c515e'

# 创建一个字典，将文件名映射到对应的特征类
file_to_class_mapping = {
    "generic_agents.gz": GenericAgents,
    "vector_map.gz": VectorMap,
    "trajectory.gz": Trajectory,
}


for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(".gz"):
            gz_file = os.path.join(root, file)
            print(f"Loading {gz_file}")
            feature_cache = FeatureCachePickle()
            feature_file = pathlib.Path(gz_file)
            # 检查文件名是否在字典中，如果是，则加载相应的特征并打印ego属性
            if file in file_to_class_mapping:
                loaded_feature = feature_cache.load_computed_feature_from_folder(feature_file, file_to_class_mapping[file])

                if file == "generic_agents.gz":
                    ego_agent_features = loaded_feature
                elif file == "vector_map.gz":
                    vector_map_data = loaded_feature
                else:
                    trajectory = loaded_feature

ego = ego_agent_features.ego