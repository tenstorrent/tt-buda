// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "third_party/json/json.hpp"
 
#include <optional>
#include <variant>
#include <tuple>
 
namespace nlohmann {
    template<>
    struct adl_serializer<std::monostate> {
        static void to_json(json& j, const std::monostate&) {
            j = nullptr;
        }
        static void from_json(const json& j, std::monostate&) {
            if (!j.is_null()) {
                throw json::type_error::create(302, std::string("type must be null, but is ") + j.type_name(), j);
            }
        }
    };

    template<typename T>
    struct adl_serializer<std::optional<T>> {
        static void to_json(json& j, const std::optional<T>& opt) {
            if (opt == std::nullopt) {
                j = nullptr;
            } else {
                j = *opt;
            }
        }

        static void from_json(const json& j, std::optional<T>& opt) {
            if (j.is_null()) {
                opt = std::nullopt;
            } else {
                opt = j.get<T>();
            }
        }
    };

    template<typename... Ts, typename T>
    static bool try_variant_type(const json& j, std::variant<Ts...>& v, T) {
        try {
            v = j.get<T>();
            return true;
        } catch (...) {
            return false;
        }
    }

    template<typename... Ts>
    struct adl_serializer<std::variant<Ts...>> {
        static void to_json(json& j, const std::variant<Ts...>& v) {
            std::visit([&](auto&& arg) { j = arg; }, v);
        }

        static void from_json(const json& j, std::variant<Ts...>& v) {
            if (j.is_null()) {
                v = std::monostate{};
            } else {
                std::apply([&](auto&&... args) {
                    ((try_variant_type(j, v, args) || ...));
                }, std::tuple<Ts...>{});
            }
        }
    };
}
 