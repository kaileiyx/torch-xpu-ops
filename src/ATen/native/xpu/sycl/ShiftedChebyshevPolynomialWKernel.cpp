#include <ATen/ATen.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/xpu/sycl/Loops.h>
#include <ATen/native/xpu/sycl/ShiftedChebyshevPolynomialKernels.h>

namespace at::native::xpu {

template <typename scalar_t>
struct ShiftedChebyshevPolynomialWFunctor {
  scalar_t operator()(scalar_t x, scalar_t n) const {
    return shifted_chebyshev_polynomial_w_forward<scalar_t>(x, n);
  }
};

void shifted_chebyshev_polynomial_w_kernel(TensorIteratorBase& iterator) {
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "shifted_chebyshev_polynomial_w_xpu", [&]() {
        ShiftedChebyshevPolynomialWFunctor<scalar_t> f;
        gpu_kernel_with_scalars(iterator, f);
      });
}

} // namespace at::native::xpu
