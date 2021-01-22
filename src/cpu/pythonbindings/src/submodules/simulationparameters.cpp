#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <complex>
#include <simulationconfig/SimulationParameters.h>

namespace py = pybind11;

void makeParametersModule(py::module_ &parentModule)
{
    py::module parametersModule = parentModule.def_submodule("parameters");

    py::class_<PhysicalParameters, std::shared_ptr<PhysicalParameters>>(parametersModule, "PhysicalParameters")
            .def(py::init())
            .def_readwrite("bulk_viscosity_factor", &PhysicalParameters::bulkViscosityFactor,
                           "The viscosity of the fluid will be multiplied with this factor to calculate its bulk viscosity. Default is 1.0")
            .def_readwrite("lattice_viscosity", &PhysicalParameters::latticeViscosity, "Lattice viscosity");

    py::class_<GridParameters, std::shared_ptr<GridParameters>>(parametersModule, "GridParameters")
            .def(py::init())
            .def_readwrite("node_distance", &GridParameters::nodeDistance)
            .def_readwrite("reference_direction_index", &GridParameters::referenceDirectionIndex)
            .def_readwrite("number_of_nodes_per_direction", &GridParameters::numberOfNodesPerDirection)
            .def_readwrite("blocks_per_direction", &GridParameters::blocksPerDirection)
            .def_readwrite("periodic_boundary_in_x1", &GridParameters::periodicBoundaryInX1)
            .def_readwrite("periodic_boundary_in_x2", &GridParameters::periodicBoundaryInX2)
            .def_readwrite("periodic_boundary_in_x3", &GridParameters::periodicBoundaryInX3)
            .def_property_readonly("bounding_box", &GridParameters::boundingBox);

    py::class_<BoundingBox, std::shared_ptr<BoundingBox>>(parametersModule, "BoundingBox")
            .def_readonly("min_x1", &BoundingBox::minX1)
            .def_readonly("min_x2", &BoundingBox::minX2)
            .def_readonly("min_x3", &BoundingBox::minX3)
            .def_readonly("max_x1", &BoundingBox::maxX1)
            .def_readonly("max_x2", &BoundingBox::maxX2)
            .def_readonly("max_x3", &BoundingBox::maxX3)
            .def("__repr__", [](BoundingBox &self)
            {
                std::ostringstream stream;
                stream << "<BoundingBox" << std::endl
                       << "min x1: " << self.minX1 << std::endl
                       << "min x2: " << self.minX2 << std::endl
                       << "min x3: " << self.minX3 << std::endl
                       << "max x1: " << self.maxX1 << std::endl
                       << "max x2: " << self.maxX2 << std::endl
                       << "max x3: " << self.maxX3 << std::endl << ">";

                return stream.str();
            });

    py::class_<RuntimeParameters, std::shared_ptr<RuntimeParameters>>(parametersModule, "RuntimeParameters")
            .def(py::init())
            .def_readwrite("number_of_timesteps", &RuntimeParameters::numberOfTimeSteps)
            .def_readwrite("timestep_log_interval", &RuntimeParameters::timeStepLogInterval)
            .def_readwrite("number_of_threads", &RuntimeParameters::numberOfThreads);

}