#pragma once
// Torch-free stand-in for <ATen/Tensor.h>, device builds only (exl3_moe.cu). The vendored exl3_moe_coop.cuh includes
// it for host function declarations that take tensors by reference; forward declarations are enough for those, and
// nothing in the device module calls them.
namespace at { class Tensor; }
namespace c10 { template <class T> class optional; }
