#pragma once

extern "C" void gpu_mat_sum(char* lv, char* rv, char* res, size_t data_size);
extern "C" void gpu_mat_sum2(char* lv, char* rv, char* res, size_t data_size, const int offset);