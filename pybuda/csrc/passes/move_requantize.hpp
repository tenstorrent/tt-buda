#pragma once

namespace tt::graphlib
{
class Graph;
}

namespace tt::passes
{

bool move_tm_through_requantize(graphlib::Graph *graph);
}
