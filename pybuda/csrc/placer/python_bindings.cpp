// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "placer/python_bindings.hpp"

#include <cstdint>

#include "graph_lib/graph.hpp"
#include "placer/placer.hpp"
#include "placer/dram_allocator.hpp"
#include "placer/lower_to_placer.hpp"


namespace tt {
inline std::optional<std::array<uint32_t, 2>> coord_as_array(std::optional<placer::Coord> const& p)
{
    if (not p)
        return std::nullopt;
    return std::array<uint32_t, 2>{p->row, p->col};
}

inline std::optional<placer::Coord> array_as_coord(std::optional<std::array<uint32_t, 2>> const& p)
{
    if (not p)
        return std::nullopt;
    return placer::Coord{.row = (*p)[0], .col = (*p)[1]};
}

void PlacerModule(py::module &m_placer) {
    py::class_<placer::Coord>(m_placer, "Coord") 
        .def_readonly("row", &placer::Coord::row)
        .def_readonly("col", &placer::Coord::col)
        .def(py::pickle(
            [](const placer::Coord& p) { // __getstate__
                return py::make_tuple(
                    p.row,
                    p.col
                );
            },
          [](py::tuple t) { // __setstate__
                if (t.size() != 2)
                {
                    throw std::runtime_error("placer::Coord: Invalid state!");
                }

                placer::Coord coord = {
                    .row = t[0].cast<std::uint32_t>(),
                    .col = t[1].cast<std::uint32_t>()
                };

                return coord;
        }));


    py::class_<placer::CoordRange>(m_placer, "CoordRange")
        .def("size_r", &placer::CoordRange::size_r)
        .def("size_c", &placer::CoordRange::size_c)
        .def_readonly("start", &placer::CoordRange::start)
        .def_readonly("end", &placer::CoordRange::end);

    py::class_<placer::PlacerConfigUpdate>(m_placer, "PlacerConfigUpdate")
        .def_readonly("op_to_chip_id_assignment", &placer::PlacerConfigUpdate::op_to_chip_id_assignment)
        .def_readonly("op_names_to_chip_break", &placer::PlacerConfigUpdate::op_names_to_chip_break)
        .def_readonly("op_names_to_epoch_break", &placer::PlacerConfigUpdate::op_names_to_epoch_break);

    using OpOverrideTypes = std::variant<bool, std::optional<uint32_t>, std::optional<std::array<uint32_t, 2>>>;
    py::class_<placer::PlacerOpOverride>(m_placer, "OpOverride")
        .def(py::init<std::optional<std::array<std::uint32_t, 2>>, bool, std::optional<uint32_t>, bool>())
        .def_readonly("grid_start", &placer::PlacerOpOverride::grid_start)
        .def_readonly("transpose_op", &placer::PlacerOpOverride::transpose_op)
        .def_readonly("temporal_epoch_break", &placer::PlacerOpOverride::temporal_epoch_break)
        .def(py::pickle(
            [](const placer::PlacerOpOverride&p) { // __getstate__
                return py::make_tuple(
                    p.grid_start,
                    p.transpose_op,
                    p.chip_id,
                    p.temporal_epoch_break
                );
            },
          [](py::tuple t) { // __setstate__
            if (t.size() != 4)
            {
                throw std::runtime_error("placer::PlacerOpOverride: Invalid state!");
            }

            placer::PlacerOpOverride p = placer::PlacerOpOverride(
                t[0].cast<std::optional<placer::Coord>>(),
                t[1].cast<bool>(),
                t[2].cast<std::optional<uint32_t>>(),
                t[3].cast<bool>()
            );

            return p;
        }))
        .def(
            "to_json",
            [](placer::PlacerOpOverride const& op_override) {
                std::unordered_map<std::string, OpOverrideTypes> d;
                d["grid_start"] = coord_as_array(op_override.grid_start);
                d["transpose_op"] = op_override.transpose_op;
                d["chip_id"] = op_override.chip_id;
                d["temporal_epoch_break"] = op_override.temporal_epoch_break;
                return d;
            })
        .def("from_json", [](std::unordered_map<std::string, OpOverrideTypes> const& d) {
            placer::PlacerOpOverride op_override;
            if (auto match = d.find("grid_start");
                match != d.end() && std::holds_alternative<std::optional<std::array<uint32_t, 2>>>(match->second))
                op_override.grid_start =
                    array_as_coord(std::get<std::optional<std::array<uint32_t, 2>>>(match->second));
            if (auto match = d.find("transpose_op"); match != d.end())
                op_override.transpose_op = std::get<bool>(match->second);
            if (auto match = d.find("chip_id");
                match != d.end() && std::holds_alternative<std::optional<uint32_t>>(match->second))
                op_override.chip_id = std::get<std::optional<uint32_t>>(match->second);
            if (auto match = d.find("temporal_epoch_break");
                match != d.end() && std::holds_alternative<bool>(match->second))
                op_override.temporal_epoch_break = std::get<bool>(match->second);
            return op_override;
        });

    py::class_<placer::OpPlacement>(m_placer, "OpPlacement")
        .def_readonly("grid_transpose", &placer::OpPlacement::grid_transpose)
        .def_readonly("placed_cores", &placer::OpPlacement::placed_cores)
        .def_readonly("chip_id", &placer::OpPlacement::chip_id)
        .def_property_readonly("epoch_id", &placer::OpPlacement::epoch_id);

    py::class_<placer::QueuePlacement>(m_placer, "QueuePlacement")
        .def_readonly("read_only", &placer::QueuePlacement::read_only)
        .def_readonly("write_only", &placer::QueuePlacement::write_only);

    py::class_<placer::PlacerSolution>(m_placer, "PlacerSolution")
        .def("chip_id", &placer::PlacerSolution::chip_id)
        .def("epoch_id", &placer::PlacerSolution::epoch_id)
        .def("temporal_epoch", py::overload_cast<uint32_t>(&placer::PlacerSolution::temporal_epoch_id, py::const_))
        .def("temporal_epoch", py::overload_cast<const std::string &>(&placer::PlacerSolution::temporal_epoch_id, py::const_))
        .def_readonly("epoch_id_to_chip", &placer::PlacerSolution::epoch_id_to_chip)
        .def_readonly("is_pipelined", &placer::PlacerSolution::is_pipelined)
        .def_readonly("name_to_op_placement", &placer::PlacerSolution::name_to_op_placement)
        .def_readonly("name_to_queue_placement", &placer::PlacerSolution::name_to_queue_placement);
    
    py::enum_<tt::placer::DRAMPlacementAlgorithm>(m_placer, "DRAMPlacementAlgorithm")
        .value("ROUND_ROBIN", tt::placer::DRAMPlacementAlgorithm::ROUND_ROBIN)
        .value("ROUND_ROBIN_FLIP_FLOP", tt::placer::DRAMPlacementAlgorithm::ROUND_ROBIN_FLIP_FLOP)
        .value("GREATEST_CAPACITY", tt::placer::DRAMPlacementAlgorithm::GREATEST_CAPACITY)
        .value("CLOSEST", tt::placer::DRAMPlacementAlgorithm::CLOSEST)
        .export_values()
        .def("to_json", [](const tt::placer::DRAMPlacementAlgorithm algorithm){
            switch (algorithm)
            {
                case tt::placer::DRAMPlacementAlgorithm::ROUND_ROBIN: return "ROUND_ROBIN";
                case tt::placer::DRAMPlacementAlgorithm::ROUND_ROBIN_FLIP_FLOP: return "ROUND_ROBIN_FLIP_FLOP";
                case tt::placer::DRAMPlacementAlgorithm::GREATEST_CAPACITY: return "GREATEST_CAPACITY";
                case tt::placer::DRAMPlacementAlgorithm::CLOSEST: return "CLOSEST";
                default: break;
            }
            throw std::runtime_error("DRAMPlacementAlgorithm::to_json with unrecognized case!");
        })
        .def("from_json", [](std::string const &encoded){
            static std::unordered_map<std::string, tt::placer::DRAMPlacementAlgorithm> decode = {
                {"ROUND_ROBIN", tt::placer::DRAMPlacementAlgorithm::ROUND_ROBIN},
                {"ROUND_ROBIN_FLIP_FLOP", tt::placer::DRAMPlacementAlgorithm::ROUND_ROBIN_FLIP_FLOP},
                {"GREATEST_CAPACITY", tt::placer::DRAMPlacementAlgorithm::GREATEST_CAPACITY},
                {"CLOSEST", tt::placer::DRAMPlacementAlgorithm::CLOSEST},
            };
            return decode.at(encoded);

        });

    m_placer.def("match_op_names_to_placer_overrides", &placer::match_op_names_to_placer_overrides);
}

} // namespace tt

