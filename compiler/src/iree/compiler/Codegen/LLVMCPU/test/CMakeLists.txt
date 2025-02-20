################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/src/iree/compiler/Codegen/LLVMCPU/test/BUILD                        #
#                                                                              #
# Use iree_cmake_extra_content from iree/build_defs.oss.bzl to add arbitrary   #
# CMake-only content.                                                          #
#                                                                              #
# To disable autogeneration for this file entirely, delete this header.        #
################################################################################

iree_add_all_subdirs()

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "aarch64_dotprod_vector_lowering.mlir"
    "aarch64_vector_lowering.mlir"
    "apply_scale_lowering.mlir"
    "assign_constant_ordinals.mlir"
    "assign_import_ordinals.mlir"
    "check_ir_before_llvm_conversion.mlir"
    "check_ir_before_llvm_conversion_not_fail_unbound.mlir"
    "convert_to_llvm.mlir"
    "data_tiling_pipeline.mlir"
    "emit_vectorization_remarks.mlir"
    "hal_executable_constants.mlir"
    "hal_interface_bindings.mlir"
    "hal_interface_constants.mlir"
    "hal_interface_workgroup_info.mlir"
    "illegal_configuration.mlir"
    "materialize_aarch64_launch_configuration.mlir"
    "materialize_riscv_launch_configuration.mlir"
    "materialize_vmvx_launch_configuration.mlir"
    "materialize_x86_64_launch_configuration.mlir"
    "peel_and_vectorize.mlir"
    "pipeline_tests.mlir"
    "split_reduction_pipeline_tests.mlir"
    "synchronize_symbol_visibility.mlir"
    "test_config_mmt4d.mlir"
    "transform_dialect_bufferize.mlir"
    "transpose_avx2_lowering.mlir"
    "triple_tiling_expert_pipeline.mlir"
    "unfused_fma.mlir"
    "vector_contract_to_arm_asm.mlir"
    "vector_contract_to_arm_intrinsics.mlir"
    "verify_linalg_transform_legality.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
