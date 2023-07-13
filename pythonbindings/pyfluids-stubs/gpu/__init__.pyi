r"""
=======================================================================================
 ____          ____    __    ______     __________   __      __       __        __
 \    \       |    |  |  |  |   _   \  |___    ___| |  |    |  |     /  \      |  |
  \    \      |    |  |  |  |  |_)   |     |  |     |  |    |  |    /    \     |  |
   \    \     |    |  |  |  |   _   /      |  |     |  |    |  |   /  /\  \    |  |
    \    \    |    |  |  |  |  | \  \      |  |     |   \__/   |  /  ____  \   |  |____
     \    \   |    |  |__|  |__|  \__\     |__|      \________/  /__/    \__\  |_______|
      \    \  |    |   ________________________________________________________________
       \    \ |    |  |  ______________________________________________________________|
        \    \|    |  |  |         __          __     __     __     ______      _______
         \         |  |  |_____   |  |        |  |   |  |   |  |   |   _  \    /  _____)
          \        |  |   _____|  |  |        |  |   |  |   |  |   |  | \  \   \_______
           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  |
            \ _____|  |__|        |________|   \_______/    |__|   |______/    (_______/

  This file is part of VirtualFluids. VirtualFluids is free software: you can
  redistribute it and/or modify it under the terms of the GNU General Public
  License as published by the Free Software Foundation, either version 3 of
  the License, or (at your option) any later version.

  VirtualFluids is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
  for more details.

  You should have received a copy of the GNU General Public License along
  with VirtualFluids (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.

! \file __init__.pyi
! \ingroup gpu
! \author Henry Korb
=======================================================================================
"""
from __future__ import annotations
from typing import Callable, ClassVar, List, Optional

from typing import overload, Union
import numpy as np
import numpy.typing as npt
import basics

from . import grid_generator as grid_generator
from . import probes as probes

class PreCollisionInteractor:
    def __init__(self, *args, **kwargs) -> None: ...


class FileCollection:
    def __init__(self, *args, **kwargs) -> None: ...

class ActuatorFarm(PreCollisionInteractor):
    def __init__(self, number_of_blades_per_turbine: int, density: float, number_of_nodes_per_blade: int, epsilon: float, level: int, delta_t: float, delta_x: float, use_host_arrays: bool) -> None: ...
    def add_turbine(self, posX: float, posY: float, posZ: float, diameter: float, omega: float, azimuth: float, yaw: float, bladeRadii: List[float]) -> None: ...
    def calc_blade_forces(self) -> None: ...
    def get_all_azimuths(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_coords_x(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_coords_x_device(self) -> int: ...
    def get_all_blade_coords_y(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_coords_y_device(self) -> int: ...
    def get_all_blade_coords_z(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_coords_z_device(self) -> int: ...
    def get_all_blade_forces_x(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_forces_x_device(self) -> int: ...
    def get_all_blade_forces_y(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_forces_y_device(self) -> int: ...
    def get_all_blade_forces_z(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_forces_z_device(self) -> int: ...
    def get_all_blade_radii(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_radii_device(self) -> int: ...
    def get_all_blade_velocities_x(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_velocities_x_device(self) -> int: ...
    def get_all_blade_velocities_y(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_velocities_y_device(self) -> int: ...
    def get_all_blade_velocities_z(self) -> npt.NDArray[np.float32]: ...
    def get_all_blade_velocities_z_device(self) -> int: ...
    def get_all_omegas(self) -> npt.NDArray[np.float32]: ...
    def get_all_turbine_pos_x(self) -> npt.NDArray[np.float32]: ...
    def get_all_turbine_pos_y(self) -> npt.NDArray[np.float32]: ...
    def get_all_turbine_pos_z(self) -> npt.NDArray[np.float32]: ...
    def get_all_yaws(self) -> npt.NDArray[np.float32]: ...
    def get_turbine_azimuth(self, turbine: int) -> float: ...
    def get_turbine_blade_coords_x(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_coords_x_device(self, turbine: int) -> int: ...
    def get_turbine_blade_coords_y(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_coords_y_device(self, turbine: int) -> int: ...
    def get_turbine_blade_coords_z(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_coords_z_device(self, turbine: int) -> int: ...
    def get_turbine_blade_forces_x(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_forces_x_device(self, turbine: int) -> int: ...
    def get_turbine_blade_forces_y(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_forces_y_device(self, turbine: int) -> int: ...
    def get_turbine_blade_forces_z(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_forces_z_device(self, turbine: int) -> int: ...
    def get_turbine_blade_radii(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_radii_device(self, turbine: int) -> int: ...
    def get_turbine_blade_velocities_x(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_velocities_x_device(self, turbine: int) -> int: ...
    def get_turbine_blade_velocities_y(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_velocities_y_device(self, turbine: int) -> int: ...
    def get_turbine_blade_velocities_z(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_blade_velocities_z_device(self, turbine: int) -> int: ...
    def get_turbine_omega(self, turbine: int) -> float: ...
    def get_turbine_pos(self, turbine: int) -> npt.NDArray[np.float32]: ...
    def get_turbine_yaw(self, turbine: int) -> float: ...
    def set_all_azimuths(self, azimuths: npt.NDArray[np.float32]) -> None: ...
    def set_all_blade_coords(self, blade_coords_x: npt.NDArray[np.float32], blade_coords_y: npt.NDArray[np.float32], blade_coords_z: npt.NDArray[np.float32]) -> None: ...
    def set_all_blade_forces(self, blade_forces_x: npt.NDArray[np.float32], blade_forces_y: npt.NDArray[np.float32], blade_forces_z: npt.NDArray[np.float32]) -> None: ...
    def set_all_blade_velocities(self, blade_velocities_x: npt.NDArray[np.float32], blade_velocities_y: npt.NDArray[np.float32], blade_velocities_z: npt.NDArray[np.float32]) -> None: ...
    def set_all_omegas(self, omegas: npt.NDArray[np.float32]) -> None: ...
    def set_all_yaws(self, yaws: npt.NDArray[np.float32]) -> None: ...
    def set_turbine_azimuth(self, turbine: int, azimuth: float) -> None: ...
    def set_turbine_blade_coords(self, turbine: int, blade_coords_x: npt.NDArray[np.float32], blade_coords_y: npt.NDArray[np.float32], blade_coords_z: npt.NDArray[np.float32]) -> None: ...
    def set_turbine_blade_forces(self, turbine: int, blade_forces_x: npt.NDArray[np.float32], blade_forces_y: npt.NDArray[np.float32], blade_forces_z: npt.NDArray[np.float32]) -> None: ...
    def set_turbine_blade_velocities(self, turbine: int, blade_velocities_x: npt.NDArray[np.float32], blade_velocities_y: npt.NDArray[np.float32], blade_velocities_z: npt.NDArray[np.float32]) -> None: ...
    def set_turbine_omega(self, turbine: int, omega: float) -> None: ...
    def set_turbine_yaw(self, turbine: int, yaw: float) -> None: ...
    @property
    def delta_t(self) -> float: ...
    @property
    def delta_x(self) -> float: ...
    @property
    def density(self) -> float: ...
    @property
    def number_of_blades_per_turbine(self) -> int: ...
    @property
    def number_of_indices(self) -> int: ...
    @property
    def number_of_grid_nodes(self) -> int: ...
    @property
    def number_of_nodes_per_blade(self) -> int: ...
    @property
    def number_of_turbines(self) -> int: ...


class BoundaryConditionFactory:
    def __init__(self) -> None: ...
    def set_geometry_boundary_condition(self, boundary_condition_type: Union[SlipBC, VelocityBC, NoSlipBC]) -> None: ...
    def set_no_slip_boundary_condition(self, boundary_condition_type: NoSlipBC) -> None: ...
    def set_precursor_boundary_condition(self, boundary_condition_type: PrecursorBC) -> None: ...
    def set_pressure_boundary_condition(self, boundary_condition_type: PressureBC) -> None: ...
    def set_slip_boundary_condition(self, boundary_condition_type: SlipBC) -> None: ...
    def set_stress_boundary_condition(self, boundary_condition_type: StressBC) -> None: ...
    def set_velocity_boundary_condition(self, boundary_condition_type: VelocityBC) -> None: ...


class MpiCommunicator:
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def get_instance() -> MpiCommunicator: ...
    def get_number_of_process(self) -> int: ...
    def get_pid(self) -> int: ...


class CudaMemoryManager:
    def __init__(self, parameter: Parameter) -> None: ...


class FileType:
    __members__: ClassVar[dict] = ...  # read-only
    VTK: ClassVar[FileType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class GridProvider:
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def make_grid_generator(builder: grid_generator.GridBuilder, para: Parameter, cuda_memory_manager: CudaMemoryManager, communicator: MpiCommunicator) -> GridProvider: ...

class MultipleGridBuilder:
    def __init__(self) -> None: ...

class GridScaling:
    __members__: ClassVar[dict] = ...  # read-only
    NotSpecified: ClassVar[GridScaling] = ...
    ScaleCompressible: ClassVar[GridScaling] = ...
    ScaleRhoSq: ClassVar[GridScaling] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class GridScalingFactory:
    def __init__(self) -> None: ...
    def set_scaling_factory(self, scaling_type) -> None: ...


class NoSlipBC:
    __members__: ClassVar[dict] = ...  # read-only
    NoSlip3rdMomentsCompressible: ClassVar[NoSlipBC] = ...
    NoSlipBounceBack: ClassVar[NoSlipBC] = ...
    NoSlipCompressible: ClassVar[NoSlipBC] = ...
    NoSlipImplicitBounceBack: ClassVar[NoSlipBC] = ...
    NoSlipIncompressible: ClassVar[NoSlipBC] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class OutputVariable:
    __members__: ClassVar[dict] = ...  # read-only
    Distributions: ClassVar[OutputVariable] = ...
    Velocities: ClassVar[OutputVariable] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class Parameter:
    @overload
    def __init__(self, number_of_processes: int, my_ID: int, config_data: Optional[basics.ConfigurationFile]) -> None: ...
    @overload
    def __init__(self, number_of_processes: int, my_ID: int) -> None: ...
    @overload
    def __init__(self, config_data: basics.ConfigurationFile) -> None: ...
    def add_actuator(self, actuator: PreCollisionInteractor) -> None: ...
    def add_probe(self, probe: PreCollisionInteractor) -> None: ...
    def get_SGS_constant(self) -> float: ...
    def get_density_ratio(self) -> float: ...
    def get_force_ratio(self) -> float: ...
    def get_is_body_force(self) -> bool: ...
    def get_output_path(self) -> str: ...
    def get_output_prefix(self) -> str: ...
    def get_velocity(self) -> float: ...
    def get_velocity_ratio(self) -> float: ...
    def get_viscosity(self) -> float: ...
    def get_viscosity_ratio(self) -> float: ...
    def set_AD_kernel(self, ad_kernel: str) -> None: ...
    def set_calc_turbulence_intensity(self, calc_velocity_and_fluctuations: bool) -> None: ...
    def set_comp_on(self, is_comp: bool) -> None: ...
    def set_density_ratio(self, density_ratio: float) -> None: ...
    def set_devices(self, devices: List[int]) -> None: ...
    def set_diff_on(self, is_diff: bool) -> None: ...
    def set_forcing(self, forcing_x: float, forcing_y: float, forcing_z: float) -> None: ...
    def set_has_wall_model_monitor(self, has_wall_monitor: bool) -> None: ...
    def set_initial_condition(self, init_func: Callable[[float, float, float], List[float]]) -> None: ...
    def set_initial_condition_log_law(self, u_star: float, z0: float, velocity_ratio: float) -> None: ...
    def set_initial_condition_perturbed_log_law(self, u_star: float, z0: float, length_x: float, length_z: float, height: float, velocity_ratio: float) -> None: ...
    def set_initial_condition_uniform(self, velocity_x: float, velocity_y: float, velocity_z: float) -> None: ...
    def set_is_body_force(self, is_body_force: bool) -> None: ...
    def set_main_kernel(self, kernel: str) -> None: ...
    def set_max_dev(self, max_dev: int) -> None: ...
    def set_max_level(self, number_of_levels: int) -> None: ...
    def set_outflow_pressure_correction_factor(self, correction_factor: float) -> None: ...
    def set_output_path(self, o_path: str) -> None: ...
    def set_output_prefix(self, o_prefix: str) -> None: ...
    def set_print_files(self, print_files: bool) -> None: ...
    def set_quadric_limiters(self, quadric_limiter_p: float, quadric_limiter_m: float, quadric_limiter_d: float) -> None: ...
    def set_temperature_BC(self, temp_bc: float) -> None: ...
    def set_temperature_init(self, temp: float) -> None: ...
    def set_timestep_end(self, tend: int) -> None: ...
    def set_timestep_of_coarse_level(self, timestep: int) -> None: ...
    def set_timestep_out(self, tout: int) -> None: ...
    def set_timestep_start_out(self, t_start_out: int) -> None: ...
    def set_use_streams(self, use_streams: bool) -> None: ...
    def set_velocity_LB(self, velocity: float) -> None: ...
    def set_velocity_ratio(self, velocity_ratio: float) -> None: ...
    def set_viscosity_LB(self, viscosity: float) -> None: ...
    def set_viscosity_ratio(self, viscosity_ratio: float) -> None: ...


class PrecursorBC:
    __members__: ClassVar[dict] = ...  # read-only
    DistributionsPrecursor: ClassVar[PrecursorBC] = ...
    NotSpecified: ClassVar[PrecursorBC] = ...
    VelocityPrecursor: ClassVar[PrecursorBC] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class PrecursorWriter(PreCollisionInteractor):
    def __init__(self, filename: str, output_path: str, x_pos: float, y_min: float, y_max: float, z_min: float, z_max: float, t_start_out: int, t_save: int, output_variable: OutputVariable, max_timesteps_per_file: int) -> None: ...


class PressureBC:
    __members__: ClassVar[dict] = ...  # read-only
    NotSpecified: ClassVar[PressureBC] = ...
    OutflowNonReflective: ClassVar[PressureBC] = ...
    OutflowNonReflectivePressureCorrection: ClassVar[PressureBC] = ...
    PressureEquilibrium: ClassVar[PressureBC] = ...
    PressureEquilibrium2: ClassVar[PressureBC] = ...
    PressureNonEquilibriumCompressible: ClassVar[PressureBC] = ...
    PressureNonEquilibriumIncompressible: ClassVar[PressureBC] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class SideType:
    __members__: ClassVar[dict] = ...  # read-only
    GEOMETRY: ClassVar[SideType] = ...
    MX: ClassVar[SideType] = ...
    MY: ClassVar[SideType] = ...
    MZ: ClassVar[SideType] = ...
    PX: ClassVar[SideType] = ...
    PY: ClassVar[SideType] = ...
    PZ: ClassVar[SideType] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class Simulation:
    @overload
    def __init__(self, parameter: Parameter, memoryManager: CudaMemoryManager, communicator, gridProvider: GridProvider, bcFactory: BoundaryConditionFactory, gridScalingFactory: GridScalingFactory) -> None: ...
    @overload
    def __init__(self, parameter: Parameter, memoryManager: CudaMemoryManager, communicator, gridProvider: GridProvider, bcFactory: BoundaryConditionFactory) -> None: ...
    @overload
    def __init__(self, parameter: Parameter, memoryManager: CudaMemoryManager, communicator, gridProvider: GridProvider, bcFactory: BoundaryConditionFactory, tmFactory: TurbulenceModelFactory, gridScalingFactory: GridScalingFactory) -> None: ...
    def addEnstrophyAnalyzer(self, t_analyse: int) -> None: ...
    def addKineticEnergyAnalyzer(self, t_analyse: int) -> None: ...
    def run(self) -> None: ...


class SlipBC:
    __members__: ClassVar[dict] = ...  # read-only
    NotSpecified: ClassVar[SlipBC] = ...
    SlipBounceBack: ClassVar[SlipBC] = ...
    SlipCompressible: ClassVar[SlipBC] = ...
    SlipCompressibleTurbulentViscosity: ClassVar[SlipBC] = ...
    SlipIncompressible: ClassVar[SlipBC] = ...
    SlipPressureCompressibleTurbulentViscosity: ClassVar[SlipBC] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class StressBC:
    __members__: ClassVar[dict] = ...  # read-only
    NotSpecified: ClassVar[StressBC] = ...
    StressBounceBack: ClassVar[StressBC] = ...
    StressCompressible: ClassVar[StressBC] = ...
    StressPressureBounceBack: ClassVar[StressBC] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class TurbulenceModel:
    __members__: ClassVar[dict] = ...  # read-only
    AMD: ClassVar[TurbulenceModel] = ...
    NONE: ClassVar[TurbulenceModel] = ...
    QR: ClassVar[TurbulenceModel] = ...
    Smagorinsky: ClassVar[TurbulenceModel] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


class TurbulenceModelFactory:
    def __init__(self, para: Parameter) -> None: ...
    def read_config_file(self, config_data: basics.ConfigurationFile) -> None: ...
    def set_model_constant(self, model_constant: float) -> None: ...
    def set_turbulence_model(self, turbulence_model: TurbulenceModel) -> None: ...


class VTKFileCollection(FileCollection):
    def __init__(self, prefix: str) -> None: ...


class VelocityBC:
    __members__: ClassVar[dict] = ...  # read-only
    NotSpecified: ClassVar[VelocityBC] = ...
    VelocityAndPressureCompressible: ClassVar[VelocityBC] = ...
    VelocityCompressible: ClassVar[VelocityBC] = ...
    VelocityIncompressible: ClassVar[VelocityBC] = ...
    VelocitySimpleBounceBackCompressible: ClassVar[VelocityBC] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...


def create_file_collection(prefix: str, type: FileType) -> FileCollection: ...
