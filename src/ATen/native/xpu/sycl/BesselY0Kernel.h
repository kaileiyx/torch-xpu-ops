#pragma once

#include <ATen/native/TensorIterator.h>

namespace at::native::xpu {

TORCH_XPU_API void bessel_y0_kernel(TensorIteratorBase& iter);

} // namespace at::native::xpu
