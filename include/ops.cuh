#pragma once
void launch_reduce_sum(const float* x, float* out, int n);
void launch_softmax(const float* x, float* y, int rows, int cols);
void launch_bias_relu(const float* x, const float* bias, float* y, int rows, int cols);
