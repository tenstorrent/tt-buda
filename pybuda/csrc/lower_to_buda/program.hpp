// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <variant>
#include <unordered_map>

#include "utils/assert.hpp"

namespace tt {

namespace program {

struct Variable {

    enum ShadowType {
        CROSS_EPOCH,
        CROSS_PROGRAM,
        NONE
    };

    std::string name_;
    int initial_value_;
    bool is_static_;
    ShadowType shadow_type_ = ShadowType::NONE;
    std::shared_ptr<Variable> shadow = nullptr;

    Variable(std::string name, bool is_static, int initial_value = 0, ShadowType shadow_type = NONE) 
        : name_(name), initial_value_(initial_value), is_static_(is_static), shadow_type_(shadow_type) 
    {
        if (needs_shadow_global_read_pointer()) {
            shadow = std::make_shared<Variable>(name_ + "_shadow", is_static_, 0, ShadowType::NONE);
        }
    }
    bool is_static() const { return is_static_; }
    std::string name() const { return name_; }
    std::string to_string() const { return "$" + name_; }

    // temporary workaround for backend
    bool needs_shadow_global_read_pointer() const { return shadow_type_ != ShadowType::NONE; }
    bool is_cross_epoch_shadow() const { return shadow_type_ == ShadowType::CROSS_EPOCH; }
    bool is_cross_program_shadow() const { return shadow_type_ == ShadowType::CROSS_PROGRAM; }
    std::shared_ptr<Variable> get_shadow_global_read_pointer() const {
        TT_ASSERT(needs_shadow_global_read_pointer());
        return shadow;
    }
};

struct Parameter {
    std::string name_;

    Parameter(std::string name) : name_(name) {}
    std::string name() const { return name_; }
    std::string to_string() const { return "$" + name_; }
};

using VariableP = std::shared_ptr<Variable>;
using VariableMap = std::unordered_map<std::string, VariableP>;
using ParameterP = std::shared_ptr<Parameter>;
using ParameterMap = std::unordered_map<std::string, ParameterP>;

struct Line {
    virtual ~Line() {}
    virtual std::string to_string(const std::string &indent) const = 0;
    virtual int indent_post_change() const { return 0; }
    virtual int indent_pre_change() const { return 0; }
};

using LineP = std::shared_ptr<Line>;

struct VarDeclaration : public Line {
    std::vector<VariableP> variables_;
    bool static_vars_;
    VarDeclaration(std::vector<VariableP> &variables, bool static_vars) : variables_(variables), static_vars_(static_vars) {}
    virtual std::string to_string(const std::string &indent) const override;
};

struct ParamDeclaration : public Line {
    std::vector<ParameterP> params_;
    ParamDeclaration(std::vector<ParameterP> &params) : params_(params) {}
    virtual std::string to_string(const std::string &indent) const override;
};

struct VarInstruction : public Line {

    enum Instruction {
        SET,
        ADD,
        INCWRAP,
        INC
    };

    Instruction opcode_;
    std::vector<std::variant<VariableP, ParameterP, int>> inputs_;
    VariableP output_;
    std::unordered_map<std::string, std::string> attributes_;

    VarInstruction(Instruction opcode, VariableP output, std::vector<std::variant<VariableP, ParameterP, int>> inputs = {},
            std::unordered_map<std::string, std::string> attributes = {}) :
        opcode_(opcode), inputs_(inputs), output_(output), attributes_(attributes) {}
    virtual std::string to_string(const std::string &indent) const override;
    std::string opcode_string() const;
};

struct QueueAttributes {
    VariableP read_ptr_global_;
    VariableP read_ptr_local_;

    // For dynamic queues, this will allocate/deallocate on specified epoch
    int epoch_allocate = -1;
    int epoch_deallocate = -1;
};

struct RamAttributes {
    VariableP read_ptr_;
    VariableP write_ptr_;
};

struct QueueSettings {
public:
    std::string name_;
    bool prologue = false;
    bool epilogue = false;
    bool global_rdptr_autoinc = false;
    bool rd_ptr_autoinc = true;
    int global_wrptr_autoinc = 0;
    std::string zero = "False";
    bool read_only = false;

private:
    enum Type {
        Queue,
        RAM,
    };

    // Usage: access through helpers. Don't expose usage of std::variant outside
    Type queue_type_;
    std::variant<QueueAttributes, RamAttributes> attributes_;

public:
    QueueSettings(std::string name, QueueAttributes queue_attributes)
        : name_(name), queue_type_(Type::Queue), attributes_(queue_attributes) {}
    QueueSettings(std::string name, RamAttributes ram_attributes)
        : name_(name), queue_type_(Type::RAM), attributes_(ram_attributes) {}

    const RamAttributes& ram_attributes() const;
    const QueueAttributes& queue_attributes() const;

    int epoch_allocate() const;
    int epoch_deallocate() const;

    std::string to_string() const;
    std::string name() const { return name_; }
};

struct Execute : public Line {
    std::string graph_name_;
    std::vector<QueueSettings> queue_settings_;

    Execute(std::string graph_name, std::vector<QueueSettings> queue_setting) :
        graph_name_(graph_name), queue_settings_(queue_setting) {}

    virtual std::string to_string(const std::string &indent) const override;
};

struct AllocateQueue : public Line {
    std::vector<QueueSettings> queue_settings_;

    AllocateQueue(std::vector<QueueSettings> queue_setting) : queue_settings_(queue_setting) {}

    virtual std::string to_string(const std::string &indent) const override;
};

struct DeallocateQueue : public Line {
    std::vector<QueueSettings> queue_settings_;

    DeallocateQueue(std::vector<QueueSettings> queue_setting) : queue_settings_(queue_setting) {}

    virtual std::string to_string(const std::string &indent) const override;
};


struct Loop : public Line {

    std::string variable_;

    Loop(std::string v) : variable_(v) {}
    virtual int indent_post_change() const override { return 1; }
    virtual std::string to_string(const std::string &indent) const override { return indent + "loop: " + variable_; }
};

struct EndLoop : public Line {

    virtual int indent_pre_change() const override { return -1; }
    virtual std::string to_string(const std::string &indent) const override { return indent + "endloop"; }
};


class Program {

    std::string name_;
    std::vector<LineP> lines_;
    VariableMap variables_;
    VariableMap static_variables_;
    ParameterMap parameters_;

    VariableP add_variable(std::string name, bool static_var, int initial_value = 0);
    VariableP add_variable(VariableP var); // add already created variable
    ParameterP add_parameter(std::string name);

public:
    Program(std::string name) : name_(name) {}
    std::string to_yaml() const;

    // Generate a standard looping program, given a number of looping variables for inputs. The provided
    // function should generate the 'execute' instructions in the core of the loop
    static Program loop_template(
        std::string program_name,
        std::vector<VariableP> queue_variables,
        std::uint64_t microbatch,
        bool has_zero_grad,
        bool is_optimizer_loop,
        bool has_cache_buffers,
        std::function<void(Program &p)> gen_execute);

    // Generate a non-looping program, generally used for the optimizer. The provided
    // function should generate the 'execute' instructions.
    static Program opt_template(
        std::string program_name,
        std::vector<VariableP> queue_variables,
        std::uint64_t microbatch,
        std::function<void(Program &p)> gen_execute);

    // Get vars, add lines
    VariableP get_var(const std::string &name) const;
    void add(LineP line) { lines_.push_back(line); }

    // Variable instructions
    void set_variable_value(VariableP var, std::variant<VariableP, ParameterP, int> value);
    void instruction_incwrap(VariableP var, VariableP increment, int wrap);
    void instruction_inc(VariableP var, VariableP increment);
    void instruction_add(VariableP var, VariableP increment);
};

std::ostream &operator<<(std::ostream &os, Program const &p);

} // namespace program
} // namespace tt
