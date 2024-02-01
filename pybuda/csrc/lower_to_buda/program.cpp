// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include <sstream>

#include "lower_to_buda/program.hpp"
#include "utils/assert.hpp"

namespace tt::program {

ParameterP Program::add_parameter(std::string name)
{
    ParameterP var = std::make_shared<Parameter>(name);
    parameters_.insert(std::make_pair(name, var));
    return var;
}


VariableP Program::add_variable(std::string name, bool static_var, int initial_value)
{
    VariableP var = std::make_shared<Variable>(name, static_var, initial_value);
    return add_variable(var);
}

VariableP Program::add_variable(VariableP var)
{
    // add already created variable
    if (var->is_static())
        static_variables_.insert(std::make_pair(var->name(), var));
    else
        variables_.insert(std::make_pair(var->name(), var));
    return var;
}

void Program::set_variable_value(VariableP var, std::variant<VariableP, ParameterP, int> value)
{
    lines_.push_back(std::make_shared<VarInstruction>(
                VarInstruction::SET, var, 
                std::vector<std::variant<VariableP, ParameterP, int>>{value},
                std::unordered_map<std::string, std::string>{}));
}

void Program::instruction_incwrap(VariableP var, VariableP increment, int wrap)
{
    lines_.push_back(std::make_shared<VarInstruction>(
                VarInstruction::INCWRAP, var, 
                std::vector<std::variant<VariableP, ParameterP, int>>{increment, wrap}, 
                std::unordered_map<std::string, std::string>{}));
}

void Program::instruction_add(VariableP var, VariableP increment)
{
    lines_.push_back(std::make_shared<VarInstruction>(
                VarInstruction::ADD, var, 
                std::vector<std::variant<VariableP, ParameterP, int>>{var, increment},
                std::unordered_map<std::string, std::string>{}));
}

void Program::instruction_inc(VariableP var, VariableP increment)
{
    lines_.push_back(std::make_shared<VarInstruction>(
                VarInstruction::INC, var, 
                std::vector<std::variant<VariableP, ParameterP, int>>{var, increment},
                std::unordered_map<std::string, std::string>{}));
}

VariableP Program::get_var(const std::string &name) const 
{
    try {
        return variables_.at(name);
    } catch (std::out_of_range &e) {
        throw std::runtime_error("Variable " + name + " doesn't exist.");
    }
}

Program Program::loop_template(
        std::string program_name,
        std::vector<VariableP> queue_variables,
        std::uint64_t microbatch,
        bool has_zero_grad,
        bool is_optimizer_loop,
        bool has_cache_buffers,
        std::function<void(Program &p)> gen_execute)
{
    if (is_optimizer_loop)
        return Program::opt_template(program_name, queue_variables, microbatch, gen_execute);

    //
    // Generate a local and global rd ptr for each queue
    Program p(program_name);

    // Create constant variables to hold 1, and microbatch
    VariableP c_zero = p.add_variable("c_zero", false, 0);
    VariableP c_one = p.add_variable("c_one", false, 1);
    VariableP c_microbatch_size = p.add_variable("c_microbatch_size", false, microbatch);


    if (has_zero_grad) {
        ParameterP p_zero_grad = p.add_parameter("p_zero_grad");
        VariableP v_zero_grad = p.add_variable("v_zero_grad", false, 0);
        p.lines_.push_back(std::make_shared<VarInstruction>(VarInstruction::Instruction::SET, v_zero_grad,
                std::vector<std::variant<VariableP, ParameterP, int>>{p_zero_grad}));
    }

    for (VariableP var : queue_variables) {
        p.add_variable(var);
        if (var->is_cross_program_shadow()) {
            VariableP shadow = var->get_shadow_global_read_pointer();
            p.add_variable(shadow);
            p.lines_.push_back(std::make_shared<VarInstruction>(VarInstruction::Instruction::SET, var,
                std::vector<std::variant<VariableP, ParameterP, int>>{shadow}));
        }
    }

    VariableP v_cache_write_index;
    ParameterP p_inner_increment;
    VariableP v_inner_increment;
    ParameterP p_outer_increment;
    VariableP v_outer_increment;
    if (has_cache_buffers) {
        ParameterP p_cache_write_index = p.add_parameter("p_cache_write_index");
        ParameterP p_inner_loop_count = p.add_parameter("p_inner_loop_count");
        p_inner_increment = p.add_parameter("p_inner_increment");
        ParameterP p_outer_loop_count = p.add_parameter("p_outer_loop_count");
        p_outer_increment = p.add_parameter("p_outer_increment");
        v_cache_write_index = p.add_variable("v_cache_write_index", false, 0);
        p.lines_.push_back(std::make_shared<VarInstruction>(VarInstruction::Instruction::SET, v_cache_write_index,
                std::vector<std::variant<VariableP, ParameterP, int>>{p_cache_write_index}));

        p.lines_.push_back(std::make_shared<Loop>(p_outer_loop_count->to_string()));
        p.lines_.push_back(std::make_shared<Loop>(p_inner_loop_count->to_string()));
    }
    else {
        ParameterP p_loop_count = p.add_parameter("p_loop_count");
        p.lines_.push_back(std::make_shared<Loop>(p_loop_count->to_string()));
    }

    for (VariableP var : queue_variables) {
        p.add_variable(var);
        if (var->is_cross_epoch_shadow()) {
            VariableP shadow = var->get_shadow_global_read_pointer();
            p.add_variable(shadow);
            p.lines_.push_back(std::make_shared<VarInstruction>(VarInstruction::Instruction::SET, var,
                std::vector<std::variant<VariableP, ParameterP, int>>{shadow}));
        }
    }

    gen_execute(p);
    if (has_cache_buffers) {
        p.lines_.push_back(std::make_shared<VarInstruction>(VarInstruction::Instruction::INC, v_cache_write_index,
                std::vector<std::variant<VariableP, ParameterP, int>>{p_inner_increment}));
        p.lines_.push_back(std::make_shared<EndLoop>());
        p.lines_.push_back(std::make_shared<VarInstruction>(VarInstruction::Instruction::INC, v_cache_write_index,
                std::vector<std::variant<VariableP, ParameterP, int>>{p_outer_increment}));
        p.lines_.push_back(std::make_shared<EndLoop>());
    }
    else {
        p.lines_.push_back(std::make_shared<EndLoop>());
    }

    return p;
    
}

// Generate a non-looping program, generally used for the optimizer. The provided
// function should generate the 'execute' instructions.
Program Program::opt_template(
        std::string program_name,
        std::vector<VariableP> queue_variables,
        std::uint64_t microbatch,
        std::function<void(Program &p)> gen_execute)
{
    // Generate a local and global rd ptr for each queue
    Program p(program_name);

    // for (VariableP var : queue_variables)
    //     p.variables_.insert(std::make_pair(var->name(), var));

    for (VariableP var : queue_variables) {
        p.add_variable(var);
        if (var->needs_shadow_global_read_pointer()) {
            VariableP shadow = var->get_shadow_global_read_pointer();
            p.add_variable(shadow);
            p.lines_.push_back(std::make_shared<VarInstruction>(VarInstruction::Instruction::SET, var,
                std::vector<std::variant<VariableP, ParameterP, int>>{shadow}));
        }
    }

    // Create constant variables to hold 1, and microbatch
    VariableP c_zero = p.add_variable("c_zero", false, 0);
    VariableP c_one = p.add_variable("c_one", false, 1);
    VariableP c_microbatch_size = p.add_variable("c_microbatch_size", false, microbatch);

    gen_execute(p);

    return p;
}

std::string VarDeclaration::to_string(const std::string &indent) const
{
    const std::string name = static_vars_ ? "staticvar" : "var";
    if (variables_.size() == 0)
        return indent + name + ": []";

    std::string ret = "";
    ret += variables_[0]->to_string() + ": " + std::to_string(variables_[0]->initial_value_);
    for (std::size_t i=1; i < variables_.size(); i++)
        ret += ", " + variables_[i]->to_string() + ": " + std::to_string(variables_[i]->initial_value_);

    return indent + name + ": {" + ret + "}";
}

std::string ParamDeclaration::to_string(const std::string &indent) const
{
    std::string ret = "";
    ret += params_[0]->to_string();
    for (std::size_t i=1; i < params_.size(); i++)
        ret += ", " + params_[i]->to_string();

    return indent + "param" + ": [" + ret + "]";
}

std::string VarInstruction::opcode_string() const
{
    switch (opcode_) {
        case VarInstruction::SET: return "set";
        case VarInstruction::ADD: return "add";
        case VarInstruction::INCWRAP: return "incwrap";
        case VarInstruction::INC: return "inc";
    }
    TT_ASSERT(false, "Unknown upcode");
    return "";
}

std::string VarInstruction::to_string(const std::string &indent) const
{
    std::string ret = indent + "varinst: [" + output_->to_string() + ", " + opcode_string();

    for (std::variant<VariableP, ParameterP, int> input : inputs_) {
        if (std::holds_alternative<VariableP>(input)) {
            ret += ", " + std::get<VariableP>(input)->to_string();
        } else if (std::holds_alternative<ParameterP>(input)) {
            ret += ", " + std::get<ParameterP>(input)->to_string();
        } else {
            ret += ", " + std::to_string(std::get<int>(input));
        }
    }


    if (!attributes_.empty()) {
        for (const auto &[key, value] : attributes_)
        {
            if (key == "value") {
                ret += ", " + value;
            } else {
                TT_ASSERT(false, "Unknown attribute", key, value);
            }
        }
    }
    ret += "]";
    return ret;
}

std::string QueueSettings::to_string() const
{
    std::string ret = name_ + ": {";
    std::vector<std::string> attrs;
    attrs.push_back("prologue: " + std::string(prologue ? "true" : "false"));
    attrs.push_back("epilogue: " + std::string(epilogue ? "true" : "false"));
    attrs.push_back("zero: " + zero);

    if (global_rdptr_autoinc) {
        attrs.push_back("global_rdptr_autoinc: " + std::to_string((int)global_rdptr_autoinc));
    }

    if (!rd_ptr_autoinc) {
        attrs.push_back("rd_ptr_autoinc: " + std::to_string((int)rd_ptr_autoinc));
    }

    if (read_only) {
        attrs.push_back("read_only: true");
    }
    if (this->queue_type_ == QueueSettings::Type::Queue) {
        const QueueAttributes& queue_attrs = this->queue_attributes();

        attrs.push_back("rd_ptr_local: " + (queue_attrs.read_ptr_local_ ? queue_attrs.read_ptr_local_->to_string() : std::string("$c_zero")));
        attrs.push_back("rd_ptr_global: " + (queue_attrs.read_ptr_global_ ? queue_attrs.read_ptr_global_->to_string() : std::string("$c_zero")));
    } else if (this->queue_type_ == QueueSettings::Type::RAM) {
        const RamAttributes& ram_attrs = this->ram_attributes();

        attrs.push_back("rd_ptr_global: " + (ram_attrs.read_ptr_ ? ram_attrs.read_ptr_->to_string() : std::string("$c_zero")));
        attrs.push_back("wr_ptr_global: " + (ram_attrs.write_ptr_ ? ram_attrs.write_ptr_->to_string() : std::string("$c_zero")));
    }

    if (global_wrptr_autoinc)
    {
        attrs.push_back("global_wrptr_autoinc: " + std::to_string(global_wrptr_autoinc));
    }

    if (attrs.size() == 0) {
        return ret + "}";
    }

    ret += attrs[0];
    for (std::size_t i=1; i < attrs.size(); i++) 
        ret += ", " + attrs[i];
    ret += "}";
    return ret;
}

int QueueSettings::epoch_allocate() const {
    if (this->queue_type_ == Type::RAM)
        return -1;

    return queue_attributes().epoch_allocate;
}

int QueueSettings::epoch_deallocate() const {
    if (this->queue_type_ == Type::RAM)
        return -1;

    return queue_attributes().epoch_deallocate;
}


const RamAttributes& QueueSettings::ram_attributes() const {
    TT_ASSERT(this->queue_type_ == Type::RAM);
    return std::get<RamAttributes>(this->attributes_);
}

const QueueAttributes& QueueSettings::queue_attributes() const {
    TT_ASSERT(this->queue_type_ == Type::Queue);
    return std::get<QueueAttributes>(this->attributes_);
}

std::string Execute::to_string(const std::string &indent) const 
{
    std::string ret = indent + "execute: {graph_name: " + graph_name_;
    if (queue_settings_.size() > 0) {
        ret += ", queue_settings: {\n";

        bool first = true;
        for (const QueueSettings &q : queue_settings_) {
            if (!first) ret += ",\n";
            first = false;
            ret += indent + "             " + q.to_string();
        }
        ret += "} }";
    }
    else {
        ret += "}";
    }

    return ret;
}

std::string qs_vec_to_str(const std::vector<QueueSettings> &qs)
{
    std::string ret = "";
    bool first = true;
    for (const QueueSettings &q : qs) {
        if (!first) ret += ", ";
        first = false;
        ret += q.name();
    }
    return ret;
}

std::string AllocateQueue::to_string(const std::string &indent) const {
    return indent + "allocate_queue: [" + qs_vec_to_str(queue_settings_) + "]";
}

std::string DeallocateQueue::to_string(const std::string &indent) const {
    return indent + "deallocate_queue: [" + qs_vec_to_str(queue_settings_) + "]";
}


std::string Program::to_yaml() const
{
    std::stringstream ss;

    ss << name_ << ":" << std::endl;

    std::vector<ParameterP> params;
    for (const auto &[name, p] : parameters_) params.push_back(p);
    if (params.size() > 0) 
        ss << "    - " << ParamDeclaration{params}.to_string("") << std::endl;

    std::vector<VariableP> vars;
    for (const auto &[name, var] : variables_) vars.push_back(var);
    if (vars.size() > 0) 
        ss << "    - " << VarDeclaration{vars, false}.to_string("") << std::endl;

    std::vector<VariableP> staticvars;
    for (const auto &[name, var] : static_variables_) staticvars.push_back(var);
    if (staticvars.size() > 0) 
        ss << "    - " << VarDeclaration{staticvars, true}.to_string("") << std::endl;
    
    int indent = 0;
    for (const LineP &line : lines_) 
    {
        indent += line->indent_pre_change();
        std::string str_indent(indent*2, ' ');
        ss << "    - " << line->to_string(str_indent) << std::endl;
        indent += line->indent_post_change();
    }
    ss << std::endl;

    return ss.str();
}

std::ostream &operator<<(std::ostream &os, Program const &p)
{
    return os << p.to_yaml();
}

} // namespace tt::program
