// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#define DEBUG_TYPE "extract-address-computation"

using namespace mlir;

namespace mlir {
namespace iree_compiler {

namespace {

struct ExtractAddressComputationPass
    : public ExtractAddressComputationBase<ExtractAddressComputationPass> {
  void runOnOperation() override;
};

}  // namespace

// Rewrite a load so that all its indices are zeros.
// E.g., %ld = memref.load %base[%off0]...[%offN]
// =>
// %new_base = subview %base[%off0,.., %offN][1,..,1][1,..,1]
// %ld = memref.load %new_base[0,..,0] :
//    memref<1x..x1xTy, strided<[1,..,1], offset: ?>>
//
// Ultimately we want to produce an affine map with the address computation.
// This will be taken care of by the expand-strided-metadata pass.
template <typename LoadLikeOp>
static void rewriteLoadLike(
    RewriterBase &rewriter, LoadLikeOp loadOp,
    std::function<Value(LoadLikeOp)> getSrcMemRef,
    std::function<LoadLikeOp(RewriterBase &, LoadLikeOp, Value /*srcMemRef*/,
                             ArrayRef<Value> /*indices*/)>
        rebuildOpFromAddressAndIndices) {
  Value srcMemRef = getSrcMemRef(loadOp);
  auto ldTy = srcMemRef.getType().cast<MemRefType>();
  unsigned loadRank = ldTy.getRank();
  // Don't waste compile time if there is nothing to rewrite.
  if (loadRank == 0) return;

  RewriterBase::InsertionGuard guard(ExtractAddressComputationPass);
  rewriter.setInsertionPoint(loadOp);
  // Create the array of ones of the right size.
  SmallVector<OpFoldResult> ones(loadRank, rewriter.getIndexAttr(1));
  Location loc = loadOp.getLoc();
  auto subview = rewriter.create<memref::SubViewOp>(
      loc, /*source=*/srcMemRef,
      /*offsets=*/getAsOpFoldResult(loadOp.getIndices()),
      // TODO: Get the right sizes for non-unary loads.
      /*sizes=*/ones, /*strides=*/ones);
  // Rewrite the load with the subview as the base pointer.
  SmallVector<Value> zeros(loadRank,
                           rewriter.create<arith::ConstantIndexOp>(loc, 0));
  LoadLikeOp newLoad = rebuildOpFromAddressAndIndices(
      rewriter, loadOp, subview.getResult(), zeros);
  rewriter.replaceOp(loadOp, newLoad.getResult());
}

static memref::LoadOp rebuildLoadOp(RewriterBase &rewriter,
                                    memref::LoadOp loadOp, Value srcMemRef,
                                    ArrayRef<Value> indices) {
  Location loc = loadOp.getLoc();
  return rewriter.create<memref::LoadOp>(loc, srcMemRef, indices);
}

static nvgpu::LdMatrixOp rebuildLdMatrixOp(RewriterBase &rewriter,
                                           nvgpu::LdMatrixOp ldMatrixOp,
                                           Value srcMemRef,
                                           ArrayRef<Value> indices) {
  Location loc = ldMatrixOp.getLoc();
  return rewriter.create<nvgpu::LdMatrixOp>(
      loc, ldMatrixOp.getResult().getType(), srcMemRef, indices,
      ldMatrixOp.getTranspose(), ldMatrixOp.getNumTiles());
}

void ExtractAddressComputationPass::runOnOperation() {
  Operation *funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  funcOp->walk([&](memref::LoadOp loadOp) {
    LLVM_DEBUG(llvm::dbgs() << "Found load:\n" << loadOp << '\n');
    rewriteLoadLike<memref::LoadOp>(
        rewriter, loadOp,
        [](memref::LoadOp loadOp) -> Value { return loadOp.getMemRef(); },
        rebuildLoadOp);
  });
  funcOp->walk([&](nvgpu::LdMatrixOp loadOp) {
    LLVM_DEBUG(llvm::dbgs() << "Found ldmatrix:\n" << loadOp << '\n');
    rewriteLoadLike<nvgpu::LdMatrixOp>(
        rewriter, loadOp,
        [](nvgpu::LdMatrixOp ldMatrixOp) -> Value {
          return ldMatrixOp.getSrcMemref();
        },
        rebuildLdMatrixOp);
  });
}

std::unique_ptr<Pass> createExtractAddressComputationPass() {
  return std::make_unique<ExtractAddressComputationPass>();
}
}  // namespace iree_compiler
}  // namespace mlir
