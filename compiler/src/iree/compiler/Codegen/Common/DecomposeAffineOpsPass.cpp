// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct DecomposeAffineOpsPass
    : public DecomposeAffineOpsBase<DecomposeAffineOpsPass> {
  void runOnOperation() override;
};

}  // namespace

void DecomposeAffineOpsPass::runOnOperation() {
  IRRewriter rewriter(&getContext());
  this->getOperation()->walk([&](AffineApplyOp op) {
    rewriter.setInsertionPoint(op);
    reorderOperandsByHoistability(rewriter, op);
    (void)decompose(rewriter, op);
  });
}

std::unique_ptr<Pass> createDecomposeAffineOpsPass() {
  return std::make_unique<DecomposeAffineOpsPass>();
}
}  // namespace iree_compiler
}  // namespace mlir
