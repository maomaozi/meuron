#pragma once

#include "stdafx.h"

std::string vec_shape_to_str(const std::vector<int>& vec);
bool checkShapeSame(const __Node *lhs, const __Node *rhs);
size_t shape_to_size(const std::vector<int>& shape);