// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <csignal>
#include <iostream>
#include <string>

#include "utils/assert.hpp"

inline void pybuda_signal_handler(int sig)
{
    std::string signal_name;
    switch (sig)
    {
        case SIGSEGV:
            signal_name = "segmentation fault";
            break;
        case SIGILL:
            signal_name = "illegal instruction";
            break;
        case SIGFPE:
            signal_name = "floating point exception";
            break;
        default:
            signal_name = std::to_string(sig);
            break;
    }

    std::cerr << "pybuda_signal_handler - signal: " << sig << " (" << signal_name << ")" << std::endl;
    std::cerr << "stacktrace: " << std::endl;

    std::vector bt = tt::assert::backtrace(100, 0);
    const std::string prefix = " --- ";
    bool in_python_section = false;
    for (const auto& frame : bt)
    {
        bool python_frame = frame.find("/python") != std::string::npos;
        if (!in_python_section && python_frame)
        {
            // We are entering the python section of the backtrace.
            in_python_section = true;
            std::cerr << prefix << std::endl;
            std::cerr << prefix << "Python frame(s)" << std::endl;
            std::cerr << prefix << std::endl;
        }

        if (python_frame)
        {
            // Skip python frames.
            continue;
        }

        in_python_section = false;
        std::cerr << prefix << frame << std::endl;
    }

    // Restore the default signal handler and raise the signal again.
    // The default signal handler will generate a core dump (if enabled).
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

class SignalHandlers
{
    public:
        SignalHandlers()
        {
            // For SIGSEGV, SIGILL and SIGFPE we register our own signal handlers,
            // to print the stacktrace before the program crashes.
            for (auto sig : {SIGSEGV, SIGILL, SIGFPE})
            {
                if (std::signal(sig, pybuda_signal_handler) == SIG_ERR)
                {
                    std::cerr << "Failed to register signal handler for signal " << sig << std::endl;
                }
            }
        }
};
