// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCOps.h"

#include "hc/IR/HCDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::hc;

#define GET_OP_CLASSES
#include "hc/IR/HCOps.cpp.inc"

// `!hc.undef` is the refinement lattice's bottom element: it is compatible
// with any refined type by construction, so verifiers that tighten over the
// iter_args / yield / result triples must treat it as a wildcard to allow
// pre-inference IR to round-trip.
static bool isUndef(Type t) { return llvm::isa<UndefType>(t); }

// Guarded terminator accessor for verifiers. An otherwise-invalid IR (e.g.
// a round-trip bug or a bad builder) could leave a non-empty block with no
// terminator at all; `Block::getTerminator()` asserts in that case, so we
// guard on `mightHaveTerminator()` first and return null for the verifier
// to turn into a diagnostic.
static Operation *tryGetTerminator(Block &block) {
  if (block.empty() || !block.mightHaveTerminator())
    return nullptr;
  return block.getTerminator();
}

//===----------------------------------------------------------------------===//
// Shared signature parse/print/verify for `hc.kernel` / `hc.func` /
// `hc.intrinsic`.
//
// All three advertise the same `@name (%a: T, ...) (-> T)?` surface so that
// the `hc_front -> hc` lowering pass can emit kernel/func/intrinsic
// parameters as real SSA block arguments. Block arg types mirror the
// `function_type` inputs one-to-one. Signatures are optional: a bare
// `hc.func @foo { ... }` keeps working while the frontend is incomplete —
// in that case the body block must have no arguments either.
//===----------------------------------------------------------------------===//

namespace {

// Parse an optional `(%arg0: T, %arg1: T) (-> T)?` signature. On success,
// populates `arguments` with zero-or-more entry-block arguments and, when a
// signature is present, stores the reconstructed `FunctionType` into
// `functionTypeAttr`. When no leading `(` is seen, both outputs are left in
// their default state so the caller can emit the legacy no-signature form.
ParseResult parseOptionalFunctionSignature(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::Argument> &arguments,
    TypeAttr &functionTypeAttr) {
  if (!succeeded(parser.parseOptionalLParen()))
    return success();
  if (failed(parser.parseOptionalRParen())) {
    if (failed(parser.parseArgumentList(arguments, AsmParser::Delimiter::None,
                                        /*allowType=*/true,
                                        /*allowAttrs=*/false)))
      return failure();
    if (failed(parser.parseRParen()))
      return failure();
  }
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    // `-> T`, `-> (T0, T1)`, or `-> ()` all round-trip; drop into the
    // parenthesised branch on an opening paren, otherwise read a single
    // type.
    if (succeeded(parser.parseOptionalLParen())) {
      if (failed(parser.parseOptionalRParen())) {
        if (failed(parser.parseTypeList(resultTypes)) ||
            failed(parser.parseRParen()))
          return failure();
      }
    } else {
      Type ty;
      if (failed(parser.parseType(ty)))
        return failure();
      resultTypes.push_back(ty);
    }
  }
  SmallVector<Type> inputTypes;
  inputTypes.reserve(arguments.size());
  for (auto &arg : arguments)
    inputTypes.push_back(arg.type);
  auto fnType = FunctionType::get(parser.getContext(), inputTypes, resultTypes);
  functionTypeAttr = TypeAttr::get(fnType);
  return success();
}

// Print the inverse of `parseOptionalFunctionSignature`. When
// `functionTypeAttr` is null we skip the signature entirely (legacy
// no-args form); when it is present we pull argument names from the entry
// block so round-trips preserve user-written `%group`/`%a`/etc.
void printOptionalFunctionSignature(OpAsmPrinter &p, Operation *op,
                                    TypeAttr functionTypeAttr, Region &body) {
  if (!functionTypeAttr)
    return;
  auto fnType = llvm::cast<FunctionType>(functionTypeAttr.getValue());
  p << '(';
  // The verifier guarantees the entry block matches `function_type.inputs`
  // whenever the op round-trips cleanly; the `size()` guard here is for
  // pretty-printing IR that is mid-construction and still unverified (the
  // legacy attribute-only declaration had no block args). Fall back to
  // type-only printing in that narrow case so the printer never crashes.
  Block &entry = body.front();
  if (entry.getNumArguments() == fnType.getNumInputs()) {
    llvm::interleaveComma(entry.getArguments(), p, [&](BlockArgument arg) {
      p.printRegionArgument(arg);
    });
  } else {
    llvm::interleaveComma(fnType.getInputs(), p,
                          [&](Type t) { p.printType(t); });
  }
  p << ')';
  ArrayRef<Type> results = fnType.getResults();
  if (results.empty())
    return;
  p << " -> ";
  if (results.size() == 1 && !llvm::isa<FunctionType>(results.front())) {
    p.printType(results.front());
  } else {
    p << '(';
    llvm::interleaveComma(results, p, [&](Type t) { p.printType(t); });
    p << ')';
  }
}

// Final glue: after the signature + any op-specific keyword pieces parse,
// finish off the op by reading the attr-dict and body region with the
// entry-block arguments we just parsed. Keeping this in one place keeps
// the three callers identical.
//
// `SizedRegion<1>` on the op demands one block, but MLIR's region parser
// produces an empty region for the `{}` source form — with or without a
// declared signature. We back-fill an entry block (carrying the signature
// arguments, when any) in that case so both the declaration form
// (`hc.intrinsic @foo(%a: T) ... {}`) and the legacy no-signature form
// (`hc.func @foo { ... }`) round-trip without tripping the region
// constraint.
ParseResult
parseSignatureTailAndBody(OpAsmParser &parser, OperationState &result,
                          ArrayRef<OpAsmParser::Argument> arguments) {
  if (failed(parser.parseOptionalAttrDictWithKeyword(result.attributes)))
    return failure();
  Region *body = result.addRegion();
  if (failed(parser.parseRegion(*body, arguments,
                                /*enableNameShadowing=*/false)))
    return failure();
  if (body->empty()) {
    Block &block = body->emplaceBlock();
    for (auto &arg : arguments) {
      Location loc =
          arg.sourceLoc.value_or(UnknownLoc::get(parser.getContext()));
      block.addArgument(arg.type, loc);
    }
  }
  return success();
}

// Verify a region-bearing signature-carrying op: when `function_type` is
// present, the entry block's arguments must match inputs one-for-one; when
// it is absent, the entry block must have no arguments. Keeps verifier
// error messages close to the op mnemonic.
LogicalResult verifyFunctionSignature(Operation *op, TypeAttr functionTypeAttr,
                                      Region &body) {
  Block &entry = body.front();
  if (!functionTypeAttr) {
    if (entry.getNumArguments() != 0)
      return op->emitOpError("body block has ")
             << entry.getNumArguments()
             << " argument(s) but no function_type is declared; add a "
                "signature like `(%arg0 : T, ...)` or remove the block "
                "arguments";
    return success();
  }
  auto fnType = llvm::cast<FunctionType>(functionTypeAttr.getValue());
  if (entry.getNumArguments() != fnType.getNumInputs())
    return op->emitOpError("body block takes ")
           << entry.getNumArguments()
           << " argument(s) but function_type declares "
           << fnType.getNumInputs() << " input(s)";
  for (auto [i, blockArg, declared] :
       llvm::enumerate(entry.getArguments(), fnType.getInputs())) {
    if (blockArg.getType() != declared)
      return op->emitOpError("body block argument #")
             << i << " type " << blockArg.getType()
             << " does not match function_type input " << declared;
  }
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// `hc.kernel`.
//===----------------------------------------------------------------------===//

ParseResult HCKernelOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr sym_nameAttr;
  if (parser.parseSymbolName(sym_nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> arguments;
  TypeAttr functionTypeAttr;
  if (parseOptionalFunctionSignature(parser, arguments, functionTypeAttr))
    return failure();
  if (functionTypeAttr)
    result.addAttribute(getFunctionTypeAttrName(result.name), functionTypeAttr);

  // `requirements = ...` predates the attr-dict form and reads more nicely
  // inline, so we keep the keyword form and elide the attr from the
  // automatic dict printing. `parseCustomAttributeWithFallback` pairs with
  // the `printStrippedAttrOrType` in the printer so the `#hc.constraints`
  // dialect prefix stays implicit in the textual IR.
  if (succeeded(parser.parseOptionalKeyword("requirements"))) {
    if (parser.parseEqual())
      return failure();
    ConstraintSetAttr req;
    if (parser.parseCustomAttributeWithFallback(req, Type{}))
      return failure();
    result.addAttribute(getRequirementsAttrName(result.name), req);
  }

  return parseSignatureTailAndBody(parser, result, arguments);
}

void HCKernelOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  printOptionalFunctionSignature(p, *this, getFunctionTypeAttr(), getBody());
  if (auto req = getRequirementsAttr()) {
    // `printStrippedAttrOrType` matches the declarative-assembly-format
    // convention and drops the `#hc.constraints` dialect prefix so the
    // textual IR stays compact (`<[...]>` instead of
    // `#hc.constraints<[...]>`).
    p << " requirements = ";
    p.printStrippedAttrOrType(req);
  }
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getSymNameAttrName(), getFunctionTypeAttrName(),
                       getRequirementsAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult HCKernelOp::verify() {
  if (auto fnTypeAttr = getFunctionTypeAttr()) {
    auto fnType = llvm::cast<FunctionType>(fnTypeAttr.getValue());
    if (!fnType.getResults().empty())
      return emitOpError(
          "kernel signatures must declare no results; kernels return via "
          "an operand-less `hc.return`");
  }
  return verifyFunctionSignature(*this, getFunctionTypeAttr(), getBody());
}

//===----------------------------------------------------------------------===//
// `hc.func`.
//===----------------------------------------------------------------------===//

ParseResult HCFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr sym_nameAttr;
  if (parser.parseSymbolName(sym_nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> arguments;
  TypeAttr functionTypeAttr;
  if (parseOptionalFunctionSignature(parser, arguments, functionTypeAttr))
    return failure();
  if (functionTypeAttr)
    result.addAttribute(getFunctionTypeAttrName(result.name), functionTypeAttr);

  if (succeeded(parser.parseOptionalKeyword("requirements"))) {
    if (parser.parseEqual())
      return failure();
    ConstraintSetAttr req;
    if (parser.parseCustomAttributeWithFallback(req, Type{}))
      return failure();
    result.addAttribute(getRequirementsAttrName(result.name), req);
  }
  if (succeeded(parser.parseOptionalKeyword("effects"))) {
    if (parser.parseEqual())
      return failure();
    EffectClassAttr eff;
    if (parser.parseCustomAttributeWithFallback(eff, Type{}))
      return failure();
    result.addAttribute(getEffectsAttrName(result.name), eff);
  }

  return parseSignatureTailAndBody(parser, result, arguments);
}

void HCFuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  printOptionalFunctionSignature(p, *this, getFunctionTypeAttr(), getBody());
  if (auto req = getRequirementsAttr()) {
    p << " requirements = ";
    p.printStrippedAttrOrType(req);
  }
  if (auto eff = getEffectsAttr()) {
    p << " effects = ";
    p.printStrippedAttrOrType(eff);
  }
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getSymNameAttrName(), getFunctionTypeAttrName(),
                       getRequirementsAttrName(), getEffectsAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult HCFuncOp::verify() {
  return verifyFunctionSignature(*this, getFunctionTypeAttr(), getBody());
}

//===----------------------------------------------------------------------===//
// `hc.intrinsic`.
//===----------------------------------------------------------------------===//

ParseResult HCIntrinsicOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr sym_nameAttr;
  if (parser.parseSymbolName(sym_nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> arguments;
  TypeAttr functionTypeAttr;
  if (parseOptionalFunctionSignature(parser, arguments, functionTypeAttr))
    return failure();
  if (functionTypeAttr)
    result.addAttribute(getFunctionTypeAttrName(result.name), functionTypeAttr);

  // `scope = #hc.scope<...>` is required on every intrinsic, so it is the
  // one non-optional keyword here.
  if (parser.parseKeyword("scope") || parser.parseEqual())
    return failure();
  ScopeAttr scope;
  if (parser.parseCustomAttributeWithFallback(scope, Type{}))
    return failure();
  result.addAttribute(getScopeAttrName(result.name), scope);

  if (succeeded(parser.parseOptionalKeyword("effects"))) {
    if (parser.parseEqual())
      return failure();
    EffectClassAttr eff;
    if (parser.parseCustomAttributeWithFallback(eff, Type{}))
      return failure();
    result.addAttribute(getEffectsAttrName(result.name), eff);
  }
  if (succeeded(parser.parseOptionalKeyword("const_kwargs"))) {
    if (parser.parseEqual())
      return failure();
    ArrayAttr kwargs;
    if (parser.parseAttribute(kwargs))
      return failure();
    result.addAttribute(getConstKwargsAttrName(result.name), kwargs);
  }

  return parseSignatureTailAndBody(parser, result, arguments);
}

void HCIntrinsicOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  printOptionalFunctionSignature(p, *this, getFunctionTypeAttr(), getBody());
  p << " scope = ";
  p.printStrippedAttrOrType(getScopeAttr());
  if (auto eff = getEffectsAttr()) {
    p << " effects = ";
    p.printStrippedAttrOrType(eff);
  }
  if (auto kwargs = getConstKwargsAttr()) {
    // `const_kwargs` is a plain builtin `ArrayAttr`, which has no dialect
    // prefix to strip; `printAttribute` renders it as `["name", ...]`
    // directly.
    p << " const_kwargs = ";
    p.printAttribute(kwargs);
  }
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getSymNameAttrName(), getFunctionTypeAttrName(),
                       getScopeAttrName(), getEffectsAttrName(),
                       getConstKwargsAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult HCIntrinsicOp::verify() {
  return verifyFunctionSignature(*this, getFunctionTypeAttr(), getBody());
}

LogicalResult HCSymbolOp::verify() {
  // Auto-generated type constraint enforces `!hc.idx` already; all that's
  // left is the "must pin an expression" rule — `!hc.idx` without an
  // expression is the inferred form of an unbound capture, not a
  // user-declared symbol binding.
  if (!llvm::cast<IdxType>(getResult().getType()).getExpr())
    return emitOpError("result must pin a symbolic expression "
                       "(e.g. `!hc.idx<\"M\">`)");
  return success();
}

LogicalResult HCForRangeOp::verify() {
  // The body block takes the induction variable plus one argument per
  // iter_args entry; result types mirror iter_inits types one-to-one.
  Block &body = getBody().front();
  unsigned expectedArgs = 1 + getIterInits().size();
  if (body.getNumArguments() != expectedArgs)
    return emitOpError("expected body block to take ")
           << expectedArgs << " arguments (induction variable + "
           << getIterInits().size() << " iter_args), got "
           << body.getNumArguments();
  if (getIterResults().size() != getIterInits().size())
    return emitOpError("iter_args (")
           << getIterInits().size() << ") and result count ("
           << getIterResults().size() << ") differ";
  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip_equal(getIterInits(), getIterResults()))) {
    auto [init, result] = pair;
    if (init.getType() != result.getType())
      return emitOpError("iter_args[")
             << idx << "] type " << init.getType() << " does not match result["
             << idx << "] type " << result.getType();
  }

  // Block arguments must line up with iter_inits one-to-one past the
  // induction variable. `!hc.undef` freely matches anything so pre-inference
  // IR round-trips; once types refine, drift between iter args and block
  // args fails verification.
  for (auto [idx, pair] : llvm::enumerate(llvm::zip_equal(
           getIterInits(), body.getArguments().drop_front(1)))) {
    auto [init, blockArg] = pair;
    Type initTy = init.getType();
    Type blockTy = blockArg.getType();
    if (initTy == blockTy || isUndef(initTy) || isUndef(blockTy))
      continue;
    return emitOpError("iter_args[")
           << idx << "] type " << initTy
           << " does not match body block argument type " << blockTy;
  }

  // The body must terminate with `hc.yield`, and its operands must match
  // the iter_results signature; mirrors `hc.if`. `!hc.undef` is accepted
  // on either side so pre-inference IR round-trips cleanly.
  auto yield = llvm::dyn_cast_or_null<HCYieldOp>(tryGetTerminator(body));
  if (!yield)
    return emitOpError("body must terminate with an `hc.yield`");
  if (yield.getValues().size() != getIterResults().size())
    return emitOpError("body yield produces ")
           << yield.getValues().size() << " values, expected "
           << getIterResults().size();
  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip_equal(yield.getValues(), getIterResults()))) {
    auto [yielded, result] = pair;
    Type yieldedTy = yielded.getType();
    Type resultTy = result.getType();
    if (yieldedTy == resultTy || isUndef(yieldedTy) || isUndef(resultTy))
      continue;
    return emitOpError("body yield[")
           << idx << "] type " << yieldedTy << " does not match result[" << idx
           << "] type " << resultTy;
  }
  return success();
}

LogicalResult HCIfOp::verify() {
  // The yield in each non-empty region must produce values matching the op's
  // result types. `!hc.undef` on either side is accepted so pre-inference IR
  // round-trips: `hc.if` is explicitly usable before yield operands have
  // concrete types, and the frontend lowering emits `!hc.undef` on one
  // branch even when the other has refined. An empty else region is fine
  // when the op produces no results, mirroring `scf.if`.
  auto checkRegion = [&](Region &region,
                         llvm::StringRef label) -> LogicalResult {
    if (region.empty())
      return success();
    // `dyn_cast_or_null` + `tryGetTerminator`: a malformed region from a
    // round-trip bug could leave a foreign terminator or no terminator at
    // all; we want a clean verifier error instead of a crash.
    auto yield =
        llvm::dyn_cast_or_null<HCYieldOp>(tryGetTerminator(region.front()));
    if (!yield)
      return emitOpError(label) << " region must terminate with an `hc.yield`";
    if (yield.getValues().size() != getNumResults())
      return emitOpError(label)
             << " yield produces " << yield.getValues().size() << " values, "
             << "expected " << getNumResults();
    for (auto [idx, pair] :
         llvm::enumerate(llvm::zip_equal(yield.getValues(), getResults()))) {
      auto [yielded, result] = pair;
      Type yieldedTy = yielded.getType();
      Type resultTy = result.getType();
      if (yieldedTy == resultTy || isUndef(yieldedTy) || isUndef(resultTy))
        continue;
      return emitOpError(label)
             << " yield[" << idx << "] type " << yieldedTy
             << " does not match result[" << idx << "] type " << resultTy;
    }
    return success();
  };
  if (failed(checkRegion(getThenRegion(), "then")))
    return failure();
  if (failed(checkRegion(getElseRegion(), "else")))
    return failure();
  if (getElseRegion().empty() && getNumResults() != 0)
    return emitOpError("must provide an `else` region when producing results");
  return success();
}

namespace {
/// Extract a concrete shape attribute from a shaped `hc` type, or `nullptr`
/// if the type does not (yet) carry one. Pre-inference IR is typically
/// `!hc.undef`, in which case rank is unknown and axis range cannot be
/// checked — later inference refines the type and picks up the check.
///
/// Fully-qualified `mlir::hc::{Buffer,Tensor,Vector}Type` are required to
/// avoid colliding with MLIR's builtin `VectorType`/`TensorType`.
ShapeAttr tryGetShape(Type t) {
  if (auto buf = llvm::dyn_cast<mlir::hc::BufferType>(t))
    return buf.getShape();
  if (auto tens = llvm::dyn_cast<mlir::hc::TensorType>(t))
    return tens.getShape();
  if (auto vec = llvm::dyn_cast<mlir::hc::VectorType>(t))
    return vec.getShape();
  return nullptr;
}

/// Walks up the parent chain and returns the first subgroup/workitem region
/// op found, or null if the op sits in the default workgroup scope. Stops
/// at the nearest `hc.kernel` / `hc.func` / `hc.intrinsic` / module-like op
/// because nested kernels/funcs re-baseline the enclosing scope.
Operation *findNarrowingScope(Operation *op) {
  Operation *cur = op->getParentOp();
  while (cur) {
    if (llvm::isa<HCSubgroupRegionOp, HCWorkitemRegionOp>(cur))
      return cur;
    if (llvm::isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(cur))
      return nullptr;
    if (cur->hasTrait<OpTrait::SymbolTable>())
      return nullptr;
    cur = cur->getParentOp();
  }
  return nullptr;
}

/// Tensor allocators (`hc.zeros`/`ones`/`full`/`empty`) are workgroup-only —
/// a tensor inside a subgroup or workitem region is a scope error, not a
/// shape error, and calling it out at verify time keeps the diagnostic
/// close to the source instead of surfacing deep in a lowering.
LogicalResult verifyTensorAllocScope(Operation *op) {
  if (Operation *narrowing = findNarrowingScope(op))
    return op->emitOpError("tensor allocator is workgroup scope only; "
                           "enclosed by ")
           << narrowing->getName() << " which narrows the scope";
  return success();
}

} // namespace

LogicalResult HCBufferDimOp::verify() {
  // Python/NumPy semantics allow negative axis indexing, but that is a
  // frontend-time convenience; the dialect form is always canonicalized
  // to a non-negative axis before landing in `hc`. The attr is signless so
  // we read the raw integer value through the stored attribute and check
  // the sign explicitly.
  if (getAxisAttr().getValue().isNegative())
    return emitOpError("axis must be non-negative, got ")
           << getAxisAttr().getValue().getSExtValue();
  // Range check requires concrete rank. Pre-inference `!hc.undef` buffers
  // have no shape metadata, so the check simply skips them; once inference
  // pins the buffer to `!hc.buffer<elem, #hc.shape<...>>` the axis has to
  // fit.
  if (auto shape = tryGetShape(getBuffer().getType())) {
    uint64_t axis = getAxisAttr().getValue().getZExtValue();
    size_t rank = shape.getDims().size();
    if (axis >= rank)
      return emitOpError("axis ")
             << axis << " is out of bounds for rank-" << rank << " buffer";
  }
  return success();
}

namespace {
/// Verify that an optional `#hc.shape` on a load/vload op matches the
/// number of index operands one-to-one. Without a shape attr, we cannot
/// check rank — leave that to inference-stage checks.
template <typename OpT>
LogicalResult verifyLoadShapeRank(OpT op,
                                  mlir::Operation::operand_range indices) {
  auto shape = op.getShapeAttr();
  if (!shape)
    return success();
  size_t expected = shape.getDims().size();
  size_t actual = indices.size();
  if (expected != actual)
    return op.emitOpError("shape rank (")
           << expected << ") does not match index count (" << actual << ")";
  return success();
}
} // namespace

LogicalResult HCLoadOp::verify() {
  return verifyLoadShapeRank(*this, getIndices());
}

LogicalResult HCVLoadOp::verify() {
  return verifyLoadShapeRank(*this, getIndices());
}

LogicalResult HCReduceOp::verify() {
  // Kind is a typed enum now; wrong spellings never reach the verifier.
  if (getAxisAttr().getValue().isNegative())
    return emitOpError("axis must be non-negative, got ")
           << getAxisAttr().getValue().getSExtValue();
  // Same rank-concrete-only story as `hc.buffer_dim`: when inference has
  // pinned the value to a shaped `hc` type, reject out-of-range axes;
  // otherwise defer to inference.
  if (auto shape = tryGetShape(getValue().getType())) {
    uint64_t axis = getAxisAttr().getValue().getZExtValue();
    size_t rank = shape.getDims().size();
    if (axis >= rank)
      return emitOpError("axis ")
             << axis << " is out of bounds for rank-" << rank << " value";
  }
  return success();
}

LogicalResult HCAsTypeOp::verify() {
  // `hc.astype` models numeric conversion and nothing else. Anything that is
  // not a builtin numeric scalar (int/float/index) is rejected so
  // `target = !hc.slice` and similar nonsense fail at verify.
  Type target = getTarget();
  if (!target.isIntOrIndexOrFloat())
    return emitOpError("target type must be a builtin integer, index, or "
                       "float type, got ")
           << target;
  // The op's declared result type must agree with `target`: scalar results
  // match it directly; tensor/vector results agree on element type; the
  // `!hc.undef` escape keeps pre-inference IR legal. Anything else is a
  // builder bug that should fail loudly instead of silently round-tripping.
  Type result = getResult().getType();
  if (isUndef(result))
    return success();
  if (result.isIntOrIndexOrFloat()) {
    if (result != target)
      return emitOpError("result type ")
             << result << " does not match target type " << target;
    return success();
  }
  auto elementOf = [](Type t) -> Type {
    if (auto tens = llvm::dyn_cast<mlir::hc::TensorType>(t))
      return tens.getElementType();
    if (auto vec = llvm::dyn_cast<mlir::hc::VectorType>(t))
      return vec.getElementType();
    return {};
  };
  if (Type elem = elementOf(result)) {
    if (elem != target)
      return emitOpError("result element type ")
             << elem << " does not match target type " << target;
    return success();
  }
  return emitOpError("result type ")
         << result
         << " must be `!hc.undef`, a builtin numeric scalar, or an "
            "`!hc.tensor`/`!hc.vector` whose element type matches target";
}

LogicalResult HCWithInactiveOp::verify() {
  // `$inactive` is a scalar literal attribute; its numeric kind must agree
  // with the masked value's element domain. The check runs only once
  // inference pins the operand to a shaped type with an element; pre-
  // inference `!hc.undef` operands skip straight to success so the op
  // remains usable out of the mechanical frontend pass.
  Type value = getValue().getType();
  if (isUndef(value) || value.isIntOrIndexOrFloat())
    return success();
  Type elem;
  if (auto tens = llvm::dyn_cast<mlir::hc::TensorType>(value))
    elem = tens.getElementType();
  else if (auto vec = llvm::dyn_cast<mlir::hc::VectorType>(value))
    elem = vec.getElementType();
  if (!elem || isUndef(elem))
    return success();
  Attribute inactive = getInactiveAttr();
  auto sameDomain = [&](Attribute a) -> bool {
    if (llvm::isa<BoolAttr>(a))
      return elem.isInteger(1);
    if (auto intAttr = llvm::dyn_cast<IntegerAttr>(a))
      return elem == intAttr.getType();
    if (auto floatAttr = llvm::dyn_cast<FloatAttr>(a))
      return elem == floatAttr.getType();
    return false;
  };
  if (!sameDomain(inactive))
    return emitOpError("inactive literal ")
           << inactive << " does not match element type " << elem;
  return success();
}

LogicalResult HCZerosOp::verify() { return verifyTensorAllocScope(*this); }
LogicalResult HCOnesOp::verify() { return verifyTensorAllocScope(*this); }
LogicalResult HCFullOp::verify() { return verifyTensorAllocScope(*this); }
LogicalResult HCEmptyOp::verify() { return verifyTensorAllocScope(*this); }

//===----------------------------------------------------------------------===//
// SymbolUserOpInterface verification for call ops.
//
// Callee existence and op kind are cheap; signature parity is also verified
// when the callee carries a `function_type` attribute. `!hc.undef` on either
// side of the parity check passes (progressive typing policy): a call site
// that has not yet been inferred, or a signature that still lists `!hc.undef`
// placeholders, should not cause spurious verify errors.
//===----------------------------------------------------------------------===//

namespace {
// Signature compatibility with progressive-typing escape hatch.
bool compatibleSigType(Type callSite, Type callee) {
  if (isUndef(callSite) || isUndef(callee))
    return true;
  return callSite == callee;
}

template <typename CallOp>
LogicalResult verifySignature(CallOp op, FunctionType fnType) {
  if (fnType.getNumInputs() != op.getArgs().size())
    return op.emitOpError("callee '@")
           << op.getCallee() << "' expects " << fnType.getNumInputs()
           << " argument(s), call site provides " << op.getArgs().size();
  if (fnType.getNumResults() != op.getResults().size())
    return op.emitOpError("callee '@")
           << op.getCallee() << "' returns " << fnType.getNumResults()
           << " result(s), call site declares " << op.getResults().size();
  for (auto [i, callSite, declared] :
       llvm::enumerate(op.getArgs().getTypes(), fnType.getInputs())) {
    if (!compatibleSigType(callSite, declared))
      return op.emitOpError("arg #")
             << i << " type " << callSite
             << " is incompatible with callee declaration " << declared;
  }
  for (auto [i, callSite, declared] :
       llvm::enumerate(op.getResults().getTypes(), fnType.getResults())) {
    if (!compatibleSigType(callSite, declared))
      return op.emitOpError("result #")
             << i << " type " << callSite
             << " is incompatible with callee declaration " << declared;
  }
  return success();
}

template <typename CalleeOp, typename CallOp>
LogicalResult verifyFlatSymbolUseAsOp(CallOp op,
                                      SymbolTableCollection &symbolTable,
                                      llvm::StringRef expectedKindLabel) {
  auto sym = symbolTable.lookupNearestSymbolFrom<CalleeOp>(op.getOperation(),
                                                           op.getCalleeAttr());
  if (!sym)
    return op.emitOpError("'")
           << op.getCallee() << "' does not reference a valid "
           << expectedKindLabel;
  if (std::optional<FunctionType> fnType = sym.getFunctionType())
    return verifySignature(op, *fnType);
  return success();
}
} // namespace

LogicalResult HCCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyFlatSymbolUseAsOp<HCFuncOp>(*this, symbolTable, "hc.func");
}

namespace {
// Translates the declared effect class into concrete side effects on the
// default resource. The callee's body is opaque; we only know "maybe reads"
// / "maybe writes" at this level, so `Pure` emits nothing, the one-sided
// classes emit the matching effect, and the unknown/absent case falls back
// to MemRead+MemWrite.
void emitEffectsForClass(
    std::optional<EffectClass> cls,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto value = cls.value_or(EffectClass::ReadWrite);
  if (value == EffectClass::Pure)
    return;
  if (value == EffectClass::Read || value == EffectClass::ReadWrite)
    effects.emplace_back(MemoryEffects::Read::get());
  if (value == EffectClass::Write || value == EffectClass::ReadWrite)
    effects.emplace_back(MemoryEffects::Write::get());
}

template <typename CalleeOp, typename CallOp>
void populateEffectsFromCallee(
    CallOp op,
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  std::optional<EffectClass> cls;
  if (auto mod = op->template getParentOfType<ModuleOp>()) {
    if (auto callee = mod.template lookupSymbol<CalleeOp>(op.getCalleeAttr()))
      cls = callee.getEffects();
  }
  emitEffectsForClass(cls, effects);
}
} // namespace

void HCCallOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  populateEffectsFromCallee<HCFuncOp>(*this, effects);
}

LogicalResult
HCCallIntrinsicOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  if (failed(verifyFlatSymbolUseAsOp<HCIntrinsicOp>(*this, symbolTable,
                                                    "hc.intrinsic")))
    return failure();
  // Callee exists and has the right kind; now enforce the const_kwarg
  // whitelist it declared, if any. Extra attributes on the call site are
  // allowed (forward-compatible with target-specific decorations).
  auto intrinsic = symbolTable.lookupNearestSymbolFrom<HCIntrinsicOp>(
      getOperation(), getCalleeAttr());
  ArrayAttr required = intrinsic.getConstKwargsAttr();
  if (!required)
    return success();
  for (Attribute entry : required) {
    llvm::StringRef name = llvm::cast<StringAttr>(entry).getValue();
    if (!(*this)->hasAttr(name))
      return emitOpError("missing required const kwarg '")
             << name << "' declared by callee '@" << getCallee() << "'";
  }
  return success();
}

void HCCallIntrinsicOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  populateEffectsFromCallee<HCIntrinsicOp>(*this, effects);
}
