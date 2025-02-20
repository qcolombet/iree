# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

################################################################################
#                                                                              #
# E2E Test Artifacts Suite                                                     #
#                                                                              #
# Includes the generated cmake files to build the e2e test artifacts.          #
# The cmake files "generated_*.cmake" are generated by the python script:      #
# "build_tools/testing/generate_cmake_e2e_test_artifacts_suite.py" and sould   #
# not be edited manually. Please read README.md for the instructions to update #
# the test artifacts.                                                          #
#                                                                              #
################################################################################

if(NOT IREE_BUILD_EXPERIMENTAL_E2E_TEST_ARTIFACTS)
  return()
endif()

################################################################################
# Defines the required variables.                                              #
################################################################################
iree_package_name(PACKAGE_NAME)
set(ROOT_ARTIFACTS_DIR "${IREE_BINARY_DIR}/e2e_test_artifacts")

include(generated_e2e_test_fetch_models.cmake)

include(generated_e2e_test_iree_artifacts.cmake)
