#include "flashinfer/norm.cuh"
#include "layers/rms_norm.h"
#include "util/check.h"
#include "util/tensor.h"
namespace yllang {

void RSMNormWithoutResidual(torch::Tensor input, torch::Tensor ouput, torch::Tensor weight, float eps) {
  SymbolicDType s_type;
  SymbolicSize s_batch_size;
  SymbolicSize s_num_heads;
  SymbolicSize s_hidden_size;
  SymbolicSize s_stride_n;
  SymbolicSize s_stride_h;
  SymbolicDevice s_device;

  if (2 == input.dim()) {
    TensorMatcher({s_batch_size, s_hidden_size})
        .WithDevice(s_device)
        .WithDType(s_type)
        .WithStride({s_stride_n, 1})
        .Verify(input)
        .Verify(ouput);

    TensorMatcher({s_hidden_size}).WithDevice(s_device).WithDType(s_type).WithStride({1}).Verify(weight);

    const auto batch_size = static_cast<uint32_t>(s_batch_size.UnWrap());
    const auto d = static_cast<uint32_t>(s_hidden_size.UnWrap());
    const auto stride = static_cast<uint32_t>(s_stride_n.UnWrap());
    CudaCheck(flashinfer::norm::RMSNorm(input.data_ptr(), weight.data_ptr(), ouput.data_ptr(), batch_size, d, stride,
                                        stride, eps));
    return;
  }

  if (3 == input.dim()) {
    TensorMatcher({s_batch_size, s_num_heads, s_hidden_size})
        .WithDevice(s_device)
        .WithDType(s_type)
        .WithStride({s_stride_n, s_stride_h, 1})
        .Verify(input)
        .Verify(ouput);

    TensorMatcher({s_hidden_size}).WithDevice(s_device).WithDType(s_type).WithStride({1}).Verify(weight);

    const auto batch_size = static_cast<uint32_t>(s_batch_size.UnWrap());
    const auto d = static_cast<uint32_t>(s_hidden_size.UnWrap());
    const auto num_heads = static_cast<uint32_t>(s_num_heads.UnWrap());
    const auto stride_n = static_cast<uint32_t>(s_stride_n.UnWrap());
    const auto stride_h = static_cast<uint32_t>(s_stride_h.UnWrap());
    CudaCheck(flashinfer::norm::QKRMSNorm(input.data_ptr(), weight.data_ptr(), ouput.data_ptr(), batch_size, num_heads,
                                          d, stride_n, stride_h, stride_n, stride_h, eps));
    return;
  }
}

void RSMNormWithResidual(torch::Tensor input, torch::Tensor residual, torch::Tensor weight, float eps) {
  SymbolicDType s_type;
  SymbolicSize s_batch_size;
  SymbolicSize s_hidden_size;
  SymbolicSize s_stride;
  SymbolicDevice s_device;

  TensorMatcher({s_batch_size, s_hidden_size})
      .WithDevice(s_device)
      .WithDType(s_type)
      .WithStride({s_stride, 1})
      .Verify(input)
      .Verify(residual);

  TensorMatcher({s_hidden_size}).WithDevice(s_device).WithDType(s_type).WithStride({1}).Verify(weight);

  const auto batch_size = static_cast<uint32_t>(s_batch_size.UnWrap());
  const auto d = static_cast<uint32_t>(s_hidden_size.UnWrap());
  const auto stride = static_cast<uint32_t>(s_stride.UnWrap());
  CudaCheck(flashinfer::norm::FusedAddRMSNorm(input.data_ptr(), residual.data_ptr(), weight.data_ptr(), batch_size, d,
                                              stride, stride, eps));
  return;
}

}  // namespace yllang