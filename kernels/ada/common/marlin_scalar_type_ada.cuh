// SPDX-License-Identifier: AGPL-3.0-only AND Apache-2.0
// Portions copied from vLLM (https://github.com/vllm-project/vllm, commit 8a36f08),
// csrc/core/scalar_type.hpp:1-363, Copyright contributors to the vLLM project, Apache-2.0.
// Changes (by Atlas contributors), listed in VLLM-MARLIN-ATTRIBUTION.md:
//   * torch-free: `<torch/headeronly/util/Exception.h>` and STD_TORCH_CHECK are replaced by
//     MARLIN_SCALAR_CHECK and MARLIN_SCALAR_CHECK_MSG, a plain `assert` (the checks only
//     guard constant construction);
//   * every constexpr member is `__host__ __device__`, so the kernel template can evaluate
//     ScalarType expressions in device code without nvcc's --expt-relaxed-constexpr, which
//     the Atlas kernel build does not pass;
//   * `id()` and `from_id()` are rewritten as explicit shifts and masks over the same six
//     fields in the same order and widths, instead of the std::pair / std::tuple / std::apply
//     fold, which is host-only library code. The ids are bit-identical to upstream's;
//   * the host-only `max()`, `min()` and `str()` members (std::variant, std::string) are
//     dropped: no kernel calls them;
//   * scalar `k*Id` constants are added beside the type constants. Device code compares
//     those integers instead of binding references to namespace-scope class objects, which
//     CUDA device code may not do.

// Marlin scalar type descriptors for the Ada (sm_89) Marlin kernels (marlin_ada.cu).
//
// A ScalarType describes a weight, activation or scale element type, including sub-byte
// integer types with a stored bias (GPTQ's uint4 with bias 8 is `kU4B8`). Its 64-bit id is a
// compile-time constant, so a type can be a template parameter of the kernel. Layout of an id,
// least significant bit first (identical to upstream vLLM):
//
//   bits  0..7   exponent            (0 for integer types)
//   bits  8..15  mantissa            (integer types: size excluding the sign bit)
//   bit   16     signed_
//   bits 17..48  bias                (two's complement, 32 bits)
//   bit   49     finite_values_only
//   bits 50..57  nan_repr
//
// Only the ids are used on the Rust side, and only for documentation: the kernels are bound
// by name (see the export list at the top of marlin_ada.cu), never by id.
#pragma once

#include <cassert>
#include <cstdint>

// Upstream's STD_TORCH_CHECK(cond, msg...) throws; here a failed check is an assert. Two
// macros instead of one variadic one, so no invocation passes an empty `__VA_ARGS__`.
#define MARLIN_SCALAR_CHECK(cond) assert(cond)
#define MARLIN_SCALAR_CHECK_MSG(cond, msg) assert((cond) && (msg))

namespace vllm {

class ScalarType {
 public:
  enum NanRepr : uint8_t {
    NAN_NONE = 0,                // nans are not supported
    NAN_IEEE_754 = 1,            // nans are: exp all 1s, mantissa not all 0s
    NAN_EXTD_RANGE_MAX_MIN = 2,  // nans are: exp all 1s, mantissa all 1s

    NAN_REPR_ID_MAX
  };

  __host__ __device__ constexpr ScalarType(uint8_t exponent, uint8_t mantissa, bool signed_,
                                           int32_t bias, bool finite_values_only = false,
                                           NanRepr nan_repr = NAN_IEEE_754)
      : exponent(exponent),
        mantissa(mantissa),
        signed_(signed_),
        bias(bias),
        finite_values_only(finite_values_only),
        nan_repr(nan_repr) {}

  __host__ __device__ static constexpr ScalarType int_(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits - 1, true, bias);
  }

  __host__ __device__ static constexpr ScalarType uint(uint8_t size_bits, int32_t bias = 0) {
    return ScalarType(0, size_bits, false, bias);
  }

  // IEEE 754 compliant floating point type
  __host__ __device__ static constexpr ScalarType float_IEEE754(uint8_t exponent,
                                                                uint8_t mantissa) {
    MARLIN_SCALAR_CHECK(mantissa > 0 && exponent > 0);
    return ScalarType(exponent, mantissa, true, 0, false, NAN_IEEE_754);
  }

  // IEEE 754 non-compliant floating point type
  __host__ __device__ static constexpr ScalarType float_(uint8_t exponent, uint8_t mantissa,
                                                         bool finite_values_only,
                                                         NanRepr nan_repr) {
    MARLIN_SCALAR_CHECK_MSG(nan_repr < NAN_REPR_ID_MAX, "Invalid NanRepr");
    MARLIN_SCALAR_CHECK(mantissa > 0 && exponent > 0);
    MARLIN_SCALAR_CHECK_MSG(nan_repr != NAN_IEEE_754,
                            "use `float_IEEE754` constructor for floating point types that "
                            "follow IEEE 754 conventions");
    return ScalarType(exponent, mantissa, true, 0, finite_values_only, nan_repr);
  }

  uint8_t const exponent;  // size of the exponent field (0 for integer types)
  uint8_t const mantissa;  // size of the mantissa field (size of the integer
                           // excluding the sign bit for integer types)
  bool const signed_;      // flag if the type supports negative numbers (i.e. has a
                           // sign bit)
  int32_t const bias;      // stored values equal value + bias,
                           // used for quantized type

  // Extra Floating point info
  bool const finite_values_only;  // i.e. no +/-inf if true
  NanRepr const nan_repr;         // how NaNs are represented
                                  // (not applicable for integer types)

  using Id = int64_t;

  // Field offsets of the id (see the file comment). The widths are upstream's
  // `member_id_field_width`: 8 bits for uint8_t and NanRepr, 1 for bool, 32 for int32_t.
  static constexpr int kExponentShift = 0;
  static constexpr int kMantissaShift = 8;
  static constexpr int kSignedShift = 16;
  static constexpr int kBiasShift = 17;
  static constexpr int kFiniteShift = 49;
  static constexpr int kNanReprShift = 50;
  static constexpr int kIdBits = 58;
  static_assert(kIdBits <= 64, "ScalarType id is too large to be stored");

  // unique id for this scalar type that can be computed at compile time for
  // c++17 template specialization
  __host__ __device__ constexpr Id id() const {
    return static_cast<Id>((uint64_t(exponent) << kExponentShift) |
                           (uint64_t(mantissa) << kMantissaShift) |
                           (uint64_t(signed_ ? 1 : 0) << kSignedShift) |
                           (uint64_t(uint32_t(bias)) << kBiasShift) |
                           (uint64_t(finite_values_only ? 1 : 0) << kFiniteShift) |
                           (uint64_t(nan_repr) << kNanReprShift));
  }

  // create a ScalarType from an id, for c++17 template specialization
  __host__ __device__ static constexpr ScalarType from_id(Id id) {
    return ScalarType(
        static_cast<uint8_t>((uint64_t(id) >> kExponentShift) & 0xffu),
        static_cast<uint8_t>((uint64_t(id) >> kMantissaShift) & 0xffu),
        ((uint64_t(id) >> kSignedShift) & 0x1u) != 0,
        static_cast<int32_t>(static_cast<uint32_t>((uint64_t(id) >> kBiasShift) & 0xffffffffu)),
        ((uint64_t(id) >> kFiniteShift) & 0x1u) != 0,
        static_cast<NanRepr>((uint64_t(id) >> kNanReprShift) & 0xffu));
  }

  __host__ __device__ constexpr int64_t size_bits() const {
    return mantissa + exponent + is_signed();
  }
  __host__ __device__ constexpr bool is_signed() const { return signed_; }
  __host__ __device__ constexpr bool is_integer() const { return exponent == 0; }
  __host__ __device__ constexpr bool is_floating_point() const { return exponent > 0; }
  __host__ __device__ constexpr bool is_ieee_754() const {
    return is_floating_point() && finite_values_only == false && nan_repr == NAN_IEEE_754;
  }
  __host__ __device__ constexpr bool has_nans() const {
    return is_floating_point() && nan_repr != NAN_NONE;
  }
  __host__ __device__ constexpr bool has_infs() const {
    return is_floating_point() && finite_values_only == false;
  }
  __host__ __device__ constexpr bool has_bias() const { return bias != 0; }

  __host__ __device__ constexpr bool operator==(ScalarType const& other) const {
    return mantissa == other.mantissa && exponent == other.exponent && bias == other.bias &&
           signed_ == other.signed_ && finite_values_only == other.finite_values_only &&
           nan_repr == other.nan_repr;
  }
};

using ScalarTypeId = ScalarType::Id;

// The size in bits of the type an id names, for device code that holds only the id.
__host__ __device__ constexpr int64_t scalar_size_bits(ScalarTypeId id) {
  return ScalarType::from_id(id).size_bits();
}

// "rust style" names generally following:
//   https://github.com/pytorch/pytorch/blob/6d9f74f0af54751311f0dd71f7e5c01a93260ab3/torch/csrc/api/include/torch/types.h#L60-L70
static inline constexpr auto kS4 = ScalarType::int_(4);
static inline constexpr auto kU4 = ScalarType::uint(4);
static inline constexpr auto kU4B8 = ScalarType::uint(4, 8);
static inline constexpr auto kS8 = ScalarType::int_(8);
static inline constexpr auto kU8 = ScalarType::uint(8);
static inline constexpr auto kU8B128 = ScalarType::uint(8, 128);

static inline constexpr auto kFE2M1f = ScalarType::float_(2, 1, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE3M2f = ScalarType::float_(3, 2, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE2M3f = ScalarType::float_(2, 3, true, ScalarType::NAN_NONE);
static inline constexpr auto kFE4M3fn =
    ScalarType::float_(4, 3, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE8M0fnu =
    ScalarType(8, 0, false, 0, true, ScalarType::NAN_EXTD_RANGE_MAX_MIN);
static inline constexpr auto kFE5M2 = ScalarType::float_IEEE754(5, 2);
static inline constexpr auto kFE8M7 = ScalarType::float_IEEE754(8, 7);
static inline constexpr auto kFE5M10 = ScalarType::float_IEEE754(5, 10);

// Fixed width style names, generally following:
//  https://github.com/pytorch/pytorch/blob/6d9f74f0af54751311f0dd71f7e5c01a93260ab3/torch/csrc/api/include/torch/types.h#L47-L57
static inline constexpr auto kInt4 = kS4;
static inline constexpr auto kUint4 = kU4;
static inline constexpr auto kUint4b8 = kU4B8;
static inline constexpr auto kInt8 = kS8;
static inline constexpr auto kUint8 = kU8;
static inline constexpr auto kUint8b128 = kU8B128;

static inline constexpr auto kFloat4_e2m1f = kFE2M1f;
static inline constexpr auto kFloat6_e3m2f = kFE3M2f;
static inline constexpr auto kFloat6_e2m3f = kFE2M3f;
static inline constexpr auto kFloat8_e4m3fn = kFE4M3fn;
static inline constexpr auto kFloat8_e5m2 = kFE5M2;
static inline constexpr auto kFloat16_e8m7 = kFE8M7;
static inline constexpr auto kFloat16_e5m10 = kFE5M10;

// colloquial names
static inline constexpr auto kHalf = kFE5M10;
static inline constexpr auto kFloat16 = kHalf;
static inline constexpr auto kBFloat16 = kFE8M7;

// Scalar ids (Atlas addition). A namespace-scope constexpr SCALAR may be read directly in
// device code; a class object may not be referenced there. Device code therefore tests
// `b_type_id == kU4Id`, never `b_type == kU4`.
static inline constexpr ScalarTypeId kS4Id = kS4.id();
static inline constexpr ScalarTypeId kU4Id = kU4.id();
static inline constexpr ScalarTypeId kU4B8Id = kU4B8.id();
static inline constexpr ScalarTypeId kS8Id = kS8.id();
static inline constexpr ScalarTypeId kU8Id = kU8.id();
static inline constexpr ScalarTypeId kU8B128Id = kU8B128.id();
static inline constexpr ScalarTypeId kFE2M1fId = kFE2M1f.id();
static inline constexpr ScalarTypeId kFE4M3fnId = kFE4M3fn.id();
static inline constexpr ScalarTypeId kFE8M0fnuId = kFE8M0fnu.id();
static inline constexpr ScalarTypeId kFloat16Id = kFloat16.id();
static inline constexpr ScalarTypeId kBFloat16Id = kBFloat16.id();

// The ids must round-trip, and the rewritten packing must give the values upstream's fold
// gives (worked by hand from the layout above).
static_assert(ScalarType::from_id(kU4B8Id) == kU4B8, "kU4B8 id round trip");
static_assert(ScalarType::from_id(kBFloat16Id) == kBFloat16, "kBFloat16 id round trip");
static_assert(kU4Id == ((int64_t(4) << 8) | (int64_t(1) << 50)), "kU4 id layout");
static_assert(kU4B8Id == ((int64_t(4) << 8) | (int64_t(8) << 17) | (int64_t(1) << 50)),
              "kU4B8 id layout");
static_assert(kBFloat16Id == (int64_t(8) | (int64_t(7) << 8) | (int64_t(1) << 16) |
                              (int64_t(1) << 50)),
              "kBFloat16 id layout");
static_assert(scalar_size_bits(kU4B8Id) == 4 && scalar_size_bits(kBFloat16Id) == 16,
              "size_bits from an id");

}  // namespace vllm
