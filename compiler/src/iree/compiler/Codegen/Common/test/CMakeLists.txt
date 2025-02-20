################################################################################
# Autogenerated by build_tools/bazel_to_cmake/bazel_to_cmake.py from           #
# compiler/src/iree/compiler/Codegen/Common/test/BUILD                         #
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
    "affinemin_canonicalization.mlir"
    "bufferize_copy_only_dispatches.mlir"
    "canonicalize_interface_load_store.mlir"
    "convert_to_destination_passing_style.mlir"
    "dead_alloc.mlir"
    "decompose_linalg_generic.mlir"
    "distribute_gpu_shared_memory.mlir"
    "erase_hal_descriptor_type.mlir"
    "flatten_memref_subspan.mlir"
    "fold_affine_min_in_distributed_loops.mlir"
    "fold_tensor_extract_op.mlir"
    "forop_canonicalization.mlir"
    "gpu_pipeline.mlir"
    "gpu_vectorization.mlir"
    "iree_comprehensive_bufferize.mlir"
    "materialize_encoding.mlir"
    "pad_dynamic_alloc.mlir"
    "reduce_bank_conflicts.mlir"
    "reductions.mlir"
    "remove_dead_allocs.mlir"
    "remove_trivial_loops.mlir"
    "swizzle_workgroup.mlir"
    "test_partitionable_loops_interface.mlir"
    "tile_and_distribute_to_workgroups.mlir"
    "tile_reduction.mlir"
    "transform_dialect_apply_pattern_op.mlir"
    "transform_ops_invalid.mlir"
    "transpose_canonicalization.mlir"
    "type_propagation.mlir"
    "vectorize_tensor_pad.mlir"
    "warp_reduction.mlir"
    "workgroup_specialization.mlir"
  TOOLS
    FileCheck
    iree-opt
  DATA
    reductions_codegen_spec.mlir
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
