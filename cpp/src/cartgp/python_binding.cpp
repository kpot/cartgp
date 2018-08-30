#include <unordered_map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>


#include "cartgp/genotype.h"

namespace py = pybind11;

using namespace cartgp;


using GPFunction = Function<py::object>;
using FunctionNamePair = std::pair<std::string, GPFunction::FunctionRef>;

std::vector<GPFunction> genotype_functions(
    const std::vector<FunctionNamePair> &functions) {
    std::vector<GPFunction> result;
    result.reserve(functions.size());
    for (const auto &item: functions) {
        result.push_back({
            item.first,
            item.second
        });
    };
    return result;
}


PYBIND11_MODULE(pycartgp, m) {

    m.doc() = R"pbdoc(Cartesian Genetic Programming)pbdoc";
    py::class_<GPFunction>(m, "GPFunction")
        .def(py::init([](const std::string &name,
                         const GPFunction::FunctionRef &func) {
                 return GPFunction{name, func};
             }),
             "Creates object associating a function with a name")
        .def(py::init([](const FunctionNamePair &name_and_func) {
                 return GPFunction{
                     name_and_func.first,
                     name_and_func.second
                 };
             }),
            "Creates object associating a function with a name")
        ;

    py::class_<SolutionInfo>(m, "SolutionInfo")
        .def(py::init<std::size_t, double>())
        .def_readonly("steps", &SolutionInfo::steps)
        .def_readonly("fitness", &SolutionInfo::fitness)
        .def("__str__", [](const SolutionInfo &info) -> std::string {
            return std::string("SolutionInfo(steps=") +
                std::to_string(info.steps) +
                ", fitness=" +  std::to_string(info.fitness) + ")";
        })
        ;

    m.def("genotype_functions", &genotype_functions,
        "Creates a mapping between a group of functions and their names");
    py::implicitly_convertible<FunctionNamePair, GPFunction>();

    py::class_<Genotype>(m, "Genotype")
        .def(py::init<const GeneInt, const GeneInt, const GeneInt,
                 const GeneInt, const GeneInt, const GeneInt, const GeneInt>(),
             ("Basic constructor. Creates a new Genotype with each node "
              "representing a function of some `arity`. Since the particular "
              "set of functions can vary slightly, the expected number of them "
              "must be specified through `num_function`. "
              "`max_levels_back` specifies how far back (how many columns back)"
              " each node can be linked to. "
              "`num_inputs` and `num_outputs` define the size of the input and "
              "the output vectors respectively. `num_rows` and `num_columns`"
              "define the layout (the grid) of the genotype."),
             py::arg("arity"), py::arg("num_functions"),
             py::arg("max_levels_back"), py::arg("num_inputs"),
             py::arg("num_outputs"), py::arg("num_rows"),
             py::arg("num_columns"))
        .def(py::init<
                const GeneInt, const GeneInt, const GeneInt,
                const GeneInt, const GeneInt>(),
             ("Creates a new Genotype with a randomly initialized genome "
              "in the most unconstrained configuration: when num_rows = 1, "
              "and num_columns = max_levels_back = depth."),
             py::arg("arity"),
             py::arg("num_functions"),
             py::arg("num_inputs"),
             py::arg("num_outputs"),
             py::arg("depth"))
        .def_property_readonly("raw", &Genotype::raw,
                               "\"Raw\" genome, as a sequence of integers")
        .def_property_readonly("arity", &Genotype::arity)
        .def_property_readonly("max_levels_back", &Genotype::max_levels_back)
        .def_property_readonly("num_inputs", &Genotype::num_inputs)
        .def_property_readonly("num_outputs", &Genotype::num_outputs)
        .def_property_readonly("num_rows", &Genotype::num_rows)
        .def_property_readonly("num_columns", &Genotype::num_columns)
        .def_property_readonly("active_nodes", &Genotype::active_nodes)
        .def_property_readonly("is_valid", &Genotype::is_valid)
        .def("evaluate", &Genotype::evaluate<py::object>,
             ("Runs given input vector through the genotype using given "
              "set of functions. Returns the output vector."),
             py::arg("functions"),
             py::arg("input_data"))
        .def("mutate", &Genotype::mutate,
             ("Mutates genome until at least `max_active_mutations` active "
              "genes were mutated. Since the mutations are random, it is not "
              "guaranteed that the mutated genes will actually get values"
              "different from the old ones."),
             py::arg("max_active_mutations") = 1)
        .def("phenotype_to_string", &Genotype::phenotype_to_string<py::object>,
             "Creates a human-readable representation of all active "
             "genes (phenotype)")
        .def("explain_outputs", &Genotype::explain_outputs<py::object>,
             ("Returns a list of string where each item contains complete "
              "expression for one of the outputs. The size of the list is thus "
              "equal to the number of outputs."))
        .def("evolve", &Genotype::evolve<py::object>,
             ("Evolves new solutions until new generations stop producing "
              "any improvements greater than the `stable_margin` margin for "
              "at least `steps_to_stabilize` iterations."),
             py::arg("functions"),
             py::arg("num_offsprings"),
             py::arg("stable_margin"),
             py::arg("steps_to_stabilize"),
             py::arg("fitness_function"))
    ;
}
