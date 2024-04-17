import numpy as np
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.common.actor_state.ego_state import EgoState, EgoStateDot
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters, get_pacifica_parameters
from nuplan.planning.simulation.controller.motion_model.abstract_motion_model import AbstractMotionModel

class BasicKinematicModel(AbstractMotionModel):
    def __init__(self, vehicle: VehicleParameters,
                 max_steering_angle: float = np.pi / 3,
                 accel_time_constant: float = 0.2,
                 steering_angle_time_constant: float = 0.05,
                 model_type: str = "CV"
                 ):
        
                self._vehicle = vehicle
                self._max_steering_angle = max_steering_angle
                self._accel_time_constant = accel_time_constant
                self._steering_angle_time_constant = steering_angle_time_constant
                self.model_type = model_type

    def get_state_dot(self, state: EgoState) -> EgoStateDot:
        """Inherited, see super class."""
        pass
    
    def _cv_model(self, state: EgoState, sampling_time: TimePoint) -> EgoState:
        pass
    
    def _ca_model(self, state: EgoState, sampling_time: TimePoint) -> EgoState:
        pass
    
    def _ckappa_model(self, state: EgoState, sampling_time: TimePoint) -> EgoState:
        pass
    
    def propagate_state(self, state: EgoState, ideal_dynamic_state: DynamicCarState, sampling_time: TimePoint) -> EgoState:
        propagating_state = state
        if self._model_type == "CV":
            propagating_state = self._cv_model(state, sampling_time)
        else:
            propagating_state = self._ca_model(state, sampling_time)
        return propagating_state
    
if __name__ == "__main__":
    vehicle = get_pacifica_parameters()
    basic_kinematic_model = BasicKinematicModel(vehicle)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 