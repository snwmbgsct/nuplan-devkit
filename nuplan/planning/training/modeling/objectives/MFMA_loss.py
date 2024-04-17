from typing import Dict, List, cast

import torch

from nuplan.planning.training.modeling.objectives.abstract_objective import AbstractObjective
from nuplan.planning.training.modeling.objectives.scenario_weight_utils import extract_scenario_type_weight
from nuplan.planning.training.modeling.types import FeaturesType, ScenarioListType, TargetsType
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory


class MFMA_Loss(AbstractObjective):
    """
    Objective that computes the MFMA loss.
    """

    def __init__(self, weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'mfma_loss'
        self._weight = weight

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["plans", "predictions", "scores", "ground_truth", "weights"]

    def compute(self, predictions: dict, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        targets_trajectory = cast(Trajectory, targets["trajectory"]) #targets_trajectory.heading.shape: torch.Size([8, 16]); xy: torch.Size([8, 16, 2])
        plans = predictions["plans"]
        predictions = predictions["predictions"]
        scores = predictions["scores"]
        cost_function_weights = predictions["cost_function_weights"]
        plan_trajs = predictions["plan_trajs"]

        predictions = predictions * weights.unsqueeze(1)
        prediction_distance = torch.norm(predictions[:, :, :, 9::10, :2] - ground_truth[:, None, 1:, 9::10, :2], dim=-1)
        plan_distance = torch.norm(plans[:, :, 9::10, :2] - ground_truth[:, None, 0, 9::10, :2], dim=-1)
        prediction_distance = prediction_distance.mean(-1).sum(-1)
        plan_distance = plan_distance.mean(-1)

        best_mode = torch.argmin(plan_distance+prediction_distance, dim=-1) 
        score_loss = F.cross_entropy(scores, best_mode)
        best_mode_plan = torch.stack([plans[i, m] for i, m in enumerate(best_mode)])
        best_mode_prediction = torch.stack([predictions[i, m] for i, m in enumerate(best_mode)])
        prediction = torch.cat([best_mode_plan.unsqueeze(1), best_mode_prediction], dim=1)

        prediction_loss: torch.tensor = 0
        for i in range(prediction.shape[1]):
            prediction_loss += F.smooth_l1_loss(prediction[:, i], ground_truth[:, i, :, :3])
            prediction_loss += F.smooth_l1_loss(prediction[:, i, -1], ground_truth[:, i, -1, :3])

        return 0.5 * prediction_loss + score_loss

class ImitationObjective(AbstractObjective):
    """
    Objective that drives the model to imitate the signals from expert behaviors/trajectories.
    """

    def __init__(self, scenario_type_loss_weighting: Dict[str, float], weight: float = 1.0):
        """
        Initializes the class

        :param name: name of the objective
        :param weight: weight contribution to the overall loss
        """
        self._name = 'imitation_objective'
        self._weight = weight
        self._fn_xy = torch.nn.modules.loss.MSELoss(reduction='none')
        self._fn_heading = torch.nn.modules.loss.L1Loss(reduction='none')
        self._scenario_type_loss_weighting = scenario_type_loss_weighting

    def name(self) -> str:
        """
        Name of the objective
        """
        return self._name

    def get_list_of_required_target_types(self) -> List[str]:
        """Implemented. See interface."""
        return ["trajectory"]

    def compute(self, predictions: FeaturesType, targets: TargetsType, scenarios: ScenarioListType) -> torch.Tensor:
        """
        Computes the objective's loss given the ground truth targets and the model's predictions
        and weights it based on a fixed weight factor.

        :param predictions: model's predictions
        :param targets: ground truth targets from the dataset
        :return: loss scalar tensor
        """
        predicted_trajectory = cast(Trajectory, predictions["trajectory"])
        targets_trajectory = cast(Trajectory, targets["trajectory"])
        loss_weights = extract_scenario_type_weight(
            scenarios, self._scenario_type_loss_weighting, device=predicted_trajectory.xy.device
        )

        broadcast_shape_xy = tuple([-1] + [1 for _ in range(predicted_trajectory.xy.dim() - 1)])
        broadcasted_loss_weights_xy = loss_weights.view(broadcast_shape_xy)
        broadcast_shape_heading = tuple([-1] + [1 for _ in range(predicted_trajectory.heading.dim() - 1)])
        broadcasted_loss_weights_heading = loss_weights.view(broadcast_shape_heading)

        weighted_xy_loss = self._fn_xy(predicted_trajectory.xy, targets_trajectory.xy) * broadcasted_loss_weights_xy
        weighted_heading_loss = (
            self._fn_heading(predicted_trajectory.heading, targets_trajectory.heading)
            * broadcasted_loss_weights_heading
        )
        # Assert that broadcasting was done correctly
        assert weighted_xy_loss.size() == predicted_trajectory.xy.size()
        assert weighted_heading_loss.size() == predicted_trajectory.heading.size()

        return self._weight * (torch.mean(weighted_xy_loss) + torch.mean(weighted_heading_loss))
