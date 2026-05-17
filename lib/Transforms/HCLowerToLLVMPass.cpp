// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Implements `-hc-lower-to-llvm`. Final lowering for the `!hc.ptr`
// family. See `doc/layouts.md` "hc.ptr and memory ops" for the contract.

#include "hc/Transforms/Passes.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCOps.h"
#include "hc/IR/HCTypes.h"

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallString.h"

namespace mlir::hc {
#define GEN_PASS_DEF_HCLOWERTOLLVM
#include "hc/Transforms/Passes.h.inc"
} // namespace mlir::hc

using namespace mlir;
using namespace mlir::hc;

namespace {

// AMDGPU canonical AS numbers (SPIR-V would get its own pass).
static unsigned llvmAddrSpaceFor(AddrSpace as) {
  switch (as) {
  case AddrSpace::Workgroup:
    return 3;
  case AddrSpace::Private:
    return 5;
  case AddrSpace::Global:
    return 1;
  }
  return 0;
}

static LLVM::LLVMPointerType convertHCPtr(MLIRContext *ctx, PtrType type) {
  return LLVM::LLVMPointerType::get(ctx, llvmAddrSpaceFor(type.getAddrSpace()));
}

static Value materializeCast(OpBuilder &builder, Type type, ValueRange inputs,
                             Location loc) {
  if (inputs.size() != 1)
    return {};
  return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
      .getResult(0);
}

class HCToLLVMTypeConverter : public TypeConverter {
public:
  explicit HCToLLVMTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) -> std::optional<Type> { return type; });
    addConversion(
        [ctx](PtrType type) -> Type { return convertHCPtr(ctx, type); });
    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }
};

// Nearest symbol-table host for workgroup globals: `gpu.module` in the
// AMDGPU pipeline, `builtin.module` in unit LITs.
static Operation *nearestSymbolTable(Operation *op) {
  Operation *cursor = op->getParentOp();
  while (cursor && !cursor->hasTrait<OpTrait::SymbolTable>())
    cursor = cursor->getParentOp();
  return cursor;
}

// workgroup -> private-linkage `llvm.mlir.global` in addrspace 3 + addressof
// (AMDGPU LDS is module-scope; no in-kernel alloca for workgroup).
// private -> `llvm.alloca` in addrspace 5 (runtime count fine).
// global -> diagnose (host owns global allocations).
struct ConvertAllocOp : public OpConversionPattern<HCAllocOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto hcPtr = cast<PtrType>(op.getResult().getType());
    LLVM::LLVMPointerType llvmPtr = convertHCPtr(op.getContext(), hcPtr);

    Type element = hcPtr.getElementType();
    if (!element)
      return op.emitOpError(
          "hc-lower-to-llvm requires a typed `!hc.ptr<..., T>` for hc.alloc; "
          "opaque allocations need an element to size the storage");
    Type llvmElem = typeConverter->convertType(element);
    if (!llvmElem)
      return op.emitOpError("failed to convert hc.alloc element type");

    Value count = adaptor.getCount();
    Location loc = op.getLoc();

    if (hcPtr.getAddrSpace() == AddrSpace::Workgroup) {
      std::optional<int64_t> staticCount = getConstantIntValue(count);
      if (!staticCount || *staticCount < 0)
        return op.emitOpError(
            "workgroup hc.alloc requires a non-negative constant count "
            "(literal-binding pass owns the constness invariant)");

      Operation *symbolTableOp = nearestSymbolTable(op);
      if (!symbolTableOp)
        return op.emitOpError(
            "workgroup hc.alloc has no enclosing symbol-table op to host the "
            "addrspace(3) global");

      Type globalType = LLVM::LLVMArrayType::get(llvmElem, *staticCount);
      LLVM::GlobalOp global;
      {
        OpBuilder::InsertionGuard guard(rewriter);
        Block &moduleBlock = symbolTableOp->getRegion(0).front();
        rewriter.setInsertionPointToStart(&moduleBlock);
        SymbolTable symbolTable(symbolTableOp);
        unsigned counter = 0;
        SmallString<64> uniqueName = SymbolTable::generateSymbolName<64>(
            "__hc_workgroup",
            [&](StringRef candidate) {
              return symbolTable.lookupSymbolIn(symbolTableOp, candidate);
            },
            counter);
        global = LLVM::GlobalOp::create(
            rewriter, loc, globalType, /*isConstant=*/false,
            LLVM::Linkage::Private, uniqueName, /*value=*/Attribute(),
            /*alignment=*/0,
            /*addrSpace=*/llvmAddrSpaceFor(AddrSpace::Workgroup));
      }
      Value addr = LLVM::AddressOfOp::create(rewriter, loc, llvmPtr,
                                             global.getSymName());
      rewriter.replaceOp(op, addr);
      return success();
    }

    if (hcPtr.getAddrSpace() == AddrSpace::Private) {
      // `arith.index_cast` (not UCC) so standard arith->LLVM completes the
      // i64 conversion -- UCCs strand when reconcile runs late.
      Value countI64 = arith::IndexCastUIOp::create(
                           rewriter, loc, rewriter.getI64Type(), count)
                           .getResult();
      Value alloca = LLVM::AllocaOp::create(rewriter, loc, llvmPtr, llvmElem,
                                            countI64, /*alignment=*/0);
      rewriter.replaceOp(op, alloca);
      return success();
    }

    return op.emitOpError(
        "hc.alloc with `global` address space has no in-kernel lowering; "
        "global pointers come from kernel arguments");
  }
};

// GEP element from `hc.ptr` element; `i8` for opaque (byte-wise stride).
struct ConvertPtrOffsetOp : public OpConversionPattern<HCPtrOffsetOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCPtrOffsetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto hcPtr = cast<PtrType>(op.getResult().getType());
    LLVM::LLVMPointerType llvmPtr = convertHCPtr(op.getContext(), hcPtr);

    Type element = hcPtr.getElementType();
    Type llvmElem =
        element ? typeConverter->convertType(element) : rewriter.getI8Type();
    if (!llvmElem)
      return op.emitOpError("failed to convert hc.ptr_offset element type");

    // `arith.index_cast` (not UCC) so standard arith->LLVM completes the
    // i64 conversion -- UCCs strand pre-reconcile.
    Value indexVal =
        arith::IndexCastUIOp::create(rewriter, op.getLoc(),
                                     rewriter.getI64Type(), adaptor.getIndex())
            .getResult();
    Value gep = LLVM::GEPOp::create(rewriter, op.getLoc(), llvmPtr, llvmElem,
                                    adaptor.getSource(), ValueRange{indexVal});
    rewriter.replaceOp(op, gep);
    return success();
  }
};

// Direct map to `llvm.load`; result type drives access width.
struct ConvertPtrLoadOp : public OpConversionPattern<HCPtrLoadOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCPtrLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type result = typeConverter->convertType(op.getResult().getType());
    if (!result)
      return op.emitOpError("failed to convert hc.ptr_load result type");
    rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, result, adaptor.getSource());
    return success();
  }
};

struct ConvertPtrStoreOp : public OpConversionPattern<HCPtrStoreOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCPtrStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                               adaptor.getDest());
    return success();
  }
};

// Vector mask -> `llvm.intr.masked.load`. Scalar `i1` -> `scf.if`
// (`llvm.intr.masked.load` rejects scalar masks).
struct ConvertPtrLoadPredOp : public OpConversionPattern<HCPtrLoadPredOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCPtrLoadPredOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type result = typeConverter->convertType(op.getResult().getType());
    if (!result)
      return op.emitOpError("failed to convert hc.ptr_load_pred result type");

    if (auto vecType = dyn_cast<mlir::VectorType>(result)) {
      Value loaded = LLVM::MaskedLoadOp::create(
                         rewriter, op.getLoc(), vecType, adaptor.getSource(),
                         adaptor.getPredicate(), adaptor.getPassthrough(),
                         /*alignment=*/rewriter.getI32IntegerAttr(0),
                         /*nontemporal=*/UnitAttr())
                         .getRes();
      rewriter.replaceOp(op, loaded);
      return success();
    }

    auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), TypeRange{result},
                                  adaptor.getPredicate(),
                                  /*withElseRegion=*/true);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      Value loaded = LLVM::LoadOp::create(rewriter, op.getLoc(), result,
                                          adaptor.getSource())
                         .getResult();
      scf::YieldOp::create(rewriter, op.getLoc(), loaded);
    }
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      scf::YieldOp::create(rewriter, op.getLoc(), adaptor.getPassthrough());
    }
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }
};

struct ConvertPtrStorePredOp : public OpConversionPattern<HCPtrStorePredOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(HCPtrStorePredOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto vecType =
            dyn_cast<mlir::VectorType>(adaptor.getValue().getType())) {
      LLVM::MaskedStoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                                  adaptor.getDest(), adaptor.getPredicate(),
                                  rewriter.getI32IntegerAttr(0));
      rewriter.eraseOp(op);
      return success();
    }

    auto ifOp = scf::IfOp::create(rewriter, op.getLoc(), TypeRange{},
                                  adaptor.getPredicate(),
                                  /*withElseRegion=*/false);
    // Default `scf.yield` already planted; insert store before it.
    {
      OpBuilder::InsertionGuard g(rewriter);
      Block &thenBlock = ifOp.getThenRegion().front();
      rewriter.setInsertionPoint(&thenBlock, thenBlock.begin());
      LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                            adaptor.getDest());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

// `gpu.func` doesn't implement `FunctionOpInterface` (own arg-attr /
// known-block-size storage). Mirror the upstream helper: convert the
// signature, apply a signature conversion to the entry block.
struct ConvertGPUFuncOpSignature : public OpConversionPattern<gpu::GPUFuncOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(gpu::GPUFuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType fnType = op.getFunctionType();
    TypeConverter::SignatureConversion conversion(fnType.getNumInputs());
    SmallVector<Type> newInputs;
    for (auto [index, input] : llvm::enumerate(fnType.getInputs())) {
      Type converted = typeConverter->convertType(input);
      if (!converted)
        return failure();
      conversion.addInputs(index, converted);
      newInputs.push_back(converted);
    }
    SmallVector<Type> newResults;
    if (failed(typeConverter->convertTypes(fnType.getResults(), newResults)))
      return failure();

    auto newType =
        FunctionType::get(rewriter.getContext(), newInputs, newResults);
    if (newType == fnType && op.getBody().empty())
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      op.setFunctionType(newType);
      if (!op.getBody().empty())
        (void)rewriter.convertRegionTypes(&op.getBody(), *typeConverter,
                                          &conversion);
    });
    return success();
  }
};

// Positional operands must follow signature conversion or verifier rejects;
// outlining keeps every other shape attr stable.
struct ConvertGPULaunchFuncOp : public OpConversionPattern<gpu::LaunchFuncOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(gpu::LaunchFuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool changed = false;
    for (auto [oldOp, newOp] :
         llvm::zip_equal(op.getKernelOperands(), adaptor.getKernelOperands())) {
      if (oldOp.getType() != newOp.getType()) {
        changed = true;
        break;
      }
    }
    if (!changed)
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      op.getKernelOperandsMutable().assign(adaptor.getKernelOperands());
    });
    return success();
  }
};

// UCC bridging host `!llvm.ptr` to `!hc.ptr<global, T>` spans addrspaces
// after HC-ptr conversion. Rewrite to `llvm.addrspacecast` (or no-op
// when addrspaces agree) -- AMDGPU expects the real cast.
struct ConvertPtrUCCToAddrSpaceCast
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
      return failure();
    if (!isa<PtrType>(op.getOutputs().front().getType()))
      return failure();
    Value input = adaptor.getInputs().front();
    auto srcPtr = dyn_cast<LLVM::LLVMPointerType>(input.getType());
    if (!srcPtr)
      return failure();
    auto dstPtr = dyn_cast<LLVM::LLVMPointerType>(
        typeConverter->convertType(op.getOutputs().front().getType()));
    if (!dstPtr)
      return failure();
    if (srcPtr == dstPtr) {
      rewriter.replaceOp(op, input);
      return success();
    }
    Value cast =
        LLVM::AddrSpaceCastOp::create(rewriter, op.getLoc(), dstPtr, input);
    rewriter.replaceOp(op, cast);
    return success();
  }
};

struct HCLowerToLLVMPass
    : public hc::impl::HCLowerToLLVMBase<HCLowerToLLVMPass> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    HCToLLVMTypeConverter converter(ctx);

    RewritePatternSet patterns(ctx);
    patterns
        .add<ConvertAllocOp, ConvertPtrOffsetOp, ConvertPtrLoadOp,
             ConvertPtrStoreOp, ConvertPtrLoadPredOp, ConvertPtrStorePredOp>(
            converter, ctx);
    // Signature conversion for any function carrying `!hc.ptr<...>`.
    // `gpu.func` lacks `FunctionOpInterface` and is handled by hand below.
    populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
    populateReturnOpTypeConversionPattern(patterns, converter);
    populateCallOpTypeConversionPattern(patterns, converter);
    patterns.add<ConvertGPUFuncOpSignature, ConvertGPULaunchFuncOp,
                 ConvertPtrUCCToAddrSpaceCast>(converter, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect, arith::ArithDialect,
                           scf::SCFDialect>();
    // UCCs bridging `!llvm.ptr` -> `!hc.ptr` must become real
    // `llvm.addrspacecast` -- `reconcile-unrealized-casts` only collapses
    // exact A->B->A chains.
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [](UnrealizedConversionCastOp op) {
          if (op.getInputs().size() != 1 || op.getOutputs().size() != 1)
            return true;
          if (!isa<PtrType>(op.getOutputs().front().getType()))
            return true;
          return !isa<LLVM::LLVMPointerType>(op.getInputs().front().getType());
        });
    // Scope: `!hc.ptr` family only.
    target.addIllegalOp<HCAllocOp, HCPtrOffsetOp, HCPtrLoadOp, HCPtrStoreOp,
                        HCPtrLoadPredOp, HCPtrStorePredOp>();
    // Legal once signature is free of `!hc.ptr<...>`.
    auto signatureLegal = [&converter](Operation *fn) {
      auto fnInterface = cast<FunctionOpInterface>(fn);
      if (!converter.isSignatureLegal(
              cast<FunctionType>(fnInterface.getFunctionType())))
        return false;
      for (Type type : fnInterface.getResultTypes())
        if (!converter.isLegal(type))
          return false;
      return true;
    };
    target.addDynamicallyLegalOp<func::FuncOp>(signatureLegal);
    target.addDynamicallyLegalOp<func::CallOp>([&converter](func::CallOp call) {
      return converter.isSignatureLegal(call.getCalleeType()) &&
             converter.isLegal(call.getOperands().getTypes());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&converter](func::ReturnOp r) {
          return converter.isLegal(r.getOperandTypes());
        });
    target.addDynamicallyLegalOp<gpu::GPUFuncOp>(
        [&converter](gpu::GPUFuncOp f) {
          return converter.isSignatureLegal(f.getFunctionType());
        });
    target.addDynamicallyLegalOp<gpu::LaunchFuncOp>(
        [&converter](gpu::LaunchFuncOp launch) {
          return converter.isLegal(launch.getKernelOperands().getTypes());
        });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
