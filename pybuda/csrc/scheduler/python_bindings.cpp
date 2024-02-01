// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "scheduler/python_bindings.hpp"
#include "scheduler/scheduler.hpp"
#include "graph_lib/graph.hpp"

#include <sstream>

using namespace tt::scheduler;

void SchedulerModule(py::module &m_scheduler) {
    py::enum_<SchedulerPolicy>(m_scheduler, "SchedulerPolicy")
        .value("Topological", SchedulerPolicy::Topological)
        .value("ModuleInputsBFS", SchedulerPolicy::ModuleInputsBFS)
        .export_values();

    py::class_<SchedulerConfig>(m_scheduler, "SchedulerConfig")
        .def(py::init<SchedulerPolicy, std::vector<std::vector<std::string>>>(), py::arg("scheduler_policy"), py::arg("scheduler_constraints"))
        .def_readwrite("policy", &SchedulerConfig::policy)
        .def_readwrite("scheduler_constraints", &SchedulerConfig::scheduler_constraints);

    m_scheduler.def(
        "policy_from_string", &policy_from_string, "Returns schedule policy from string", py::arg("schedule_policy_str"));

    m_scheduler.def(
        "run_scheduler", &run_scheduler, py::arg("scheduler_config"), py::arg("graph")
    );
}

