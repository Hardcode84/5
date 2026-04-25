// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace mlir::hc;

#include "hc/IR/HCOpsInterfaces.cpp.inc"

#define GET_OP_CLASSES
#include "hc/IR/HCOps.cpp.inc"

// `!hc.undef` is the refinement lattice's bottom element: it is compatible
// with any refined type by construction, so verifiers that tighten over the
// iter_args / yield / result triples must treat it as a wildcard to allow
// pre-inference IR to round-trip.
static bool isUndef(Type t) { return llvm::isa<UndefType>(t); }

// Signature-compat check with the progressive-typing escape. Kept near
// `isUndef` so every verifier that walks op↔signature pairs can share it.
static bool compatibleSigType(Type callSite, Type callee) {
  if (isUndef(callSite) || isUndef(callee))
    return true;
  return callSite == callee;
}

static bool compatibleBranchType(Type source, Type dest) {
  if (compatibleSigType(source, dest))
    return true;
  if (auto joinable = dyn_cast<HCJoinableTypeInterface>(source))
    if (joinable.joinHCType(dest))
      return true;
  if (auto joinable = dyn_cast<HCJoinableTypeInterface>(dest))
    if (joinable.joinHCType(source))
      return true;
  return false;
}

ArrayAttr mlir::hc::filterIntrinsicOperandParameters(ArrayAttr parameters,
                                                     ArrayAttr constKwargs) {
  if (!parameters || !constKwargs || constKwargs.empty())
    return parameters;

  llvm::SmallDenseSet<StringRef> skip;
  for (Attribute kw : constKwargs) {
    auto kwName = dyn_cast<StringAttr>(kw);
    if (kwName)
      skip.insert(kwName.getValue());
  }

  SmallVector<Attribute> filtered;
  filtered.reserve(parameters.size());
  for (Attribute parameter : parameters) {
    auto name = dyn_cast<StringAttr>(parameter);
    if (name && !skip.contains(name.getValue()))
      filtered.push_back(name);
  }
  return ArrayAttr::get(parameters.getContext(), filtered);
}

FunctionType mlir::hc::getIntrinsicOperandFunctionType(
    ArrayAttr parameters, ArrayAttr constKwargs, TypeRange resultTypes,
    Type uniformOperandType) {
  ArrayAttr operands =
      filterIntrinsicOperandParameters(parameters, constKwargs);
  SmallVector<Type> inputTypes;
  if (operands) {
    inputTypes.reserve(operands.size());
    inputTypes.append(operands.size(), uniformOperandType);
  }
  return FunctionType::get(parameters.getContext(), inputTypes, resultTypes);
}

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
//
// MLIR's region parser for the `{}` source form produces an empty region
// regardless of whether a signature was declared, so
// `parseSignatureTailAndBody` back-fills an entry block below to keep
// `SizedRegion<1>` happy. The printer and verifier guard on `body.empty()` so a
// malformed in-memory op emits a diagnostic instead of crashing on
// `body.front()`.
//===----------------------------------------------------------------------===//

// Parse an optional `(%arg0: T, %arg1: T) (-> T)?` signature. On success,
// populates `arguments` with zero-or-more entry-block arguments and, when a
// signature is present, stores the reconstructed `FunctionType` into
// `functionTypeAttr`. When no leading `(` is seen, both outputs are left in
// their default state so the caller can emit the legacy no-signature form.
static ParseResult parseOptionalFunctionSignature(
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
static void printOptionalFunctionSignature(OpAsmPrinter &p, Operation *op,
                                           TypeAttr functionTypeAttr,
                                           Region &body) {
  if (!functionTypeAttr)
    return;
  auto fnType = llvm::cast<FunctionType>(functionTypeAttr.getValue());
  p << '(';
  // The verifier guarantees a non-empty entry block whose args match
  // `function_type.inputs` whenever the op round-trips cleanly. Mid-
  // construction IR can violate either invariant; fall back to type-only
  // printing in that narrow case so the printer never dereferences a
  // missing block.
  if (!body.empty() &&
      body.front().getNumArguments() == fnType.getNumInputs()) {
    llvm::interleaveComma(
        body.front().getArguments(), p,
        [&](BlockArgument arg) { p.printRegionArgument(arg); });
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

// Glue for kernel/func/intrinsic parsers: read attr-dict + body region with
// the entry-block arguments the caller already parsed. See file-level
// rationale above for the back-fill on empty-region `{}` bodies.
static ParseResult
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
static LogicalResult verifyFunctionSignature(Operation *op,
                                             TypeAttr functionTypeAttr,
                                             Region &body) {
  // `SizedRegion<1>` is enforced by ODS before custom verify fires, but a
  // badly built in-memory op could still land here with an empty region;
  // emit a diagnostic rather than let `body.front()` fire an assertion.
  if (body.empty())
    return op->emitOpError("expected a body region with an entry block");
  Block &entry = body.front();
  if (!functionTypeAttr) {
    if (entry.getNumArguments() != 0)
      return op->emitOpError("body block takes ")
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

Region *HCFuncOp::getCallableRegion() { return &getBody(); }

ArrayRef<Type> HCFuncOp::getArgumentTypes() {
  if (std::optional<FunctionType> fnType = getFunctionType())
    return fnType->getInputs();
  return {};
}

ArrayRef<Type> HCFuncOp::getResultTypes() {
  if (std::optional<FunctionType> fnType = getFunctionType())
    return fnType->getResults();
  return {};
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
  if (succeeded(parser.parseOptionalKeyword("parameters"))) {
    if (parser.parseEqual())
      return failure();
    ArrayAttr parameters;
    if (parser.parseAttribute(parameters))
      return failure();
    result.addAttribute(getParametersAttrName(result.name), parameters);
  }
  if (succeeded(parser.parseOptionalKeyword("keyword_only"))) {
    if (parser.parseEqual())
      return failure();
    ArrayAttr keywordOnly;
    if (parser.parseAttribute(keywordOnly))
      return failure();
    result.addAttribute(getKeywordOnlyAttrName(result.name), keywordOnly);
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
  if (auto parameters = getParametersAttr()) {
    p << " parameters = ";
    p.printAttribute(parameters);
  }
  if (auto keywordOnly = getKeywordOnlyAttr()) {
    p << " keyword_only = ";
    p.printAttribute(keywordOnly);
  }
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getSymNameAttrName(), getFunctionTypeAttrName(),
                       getScopeAttrName(), getEffectsAttrName(),
                       getConstKwargsAttrName(), getParametersAttrName(),
                       getKeywordOnlyAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult HCIntrinsicOp::verify() {
  if (failed(verifyFunctionSignature(*this, getFunctionTypeAttr(), getBody())))
    return failure();

  ArrayAttr parameters = getParametersAttr();
  TypeAttr fnTypeAttr = getFunctionTypeAttr();
  if (!parameters) {
    if (ArrayAttr constKwargs = getConstKwargsAttr())
      return emitOpError("const_kwargs requires parameters to declare the "
                         "full intrinsic parameter order");
    if (ArrayAttr keywordOnly = getKeywordOnlyAttr())
      return emitOpError("keyword_only requires parameters to declare the "
                         "full intrinsic parameter order");
    if (fnTypeAttr) {
      auto fnType = llvm::cast<FunctionType>(fnTypeAttr.getValue());
      if (fnType.getNumInputs() != 0)
        return emitOpError("function_type with inputs requires parameters to "
                           "name the intrinsic operand order");
    }
    return success();
  }

  llvm::SmallDenseSet<StringRef> declared;
  for (auto [idx, parameter] : llvm::enumerate(parameters)) {
    auto name = dyn_cast<StringAttr>(parameter);
    if (!name)
      return emitOpError("parameters entry at index ")
             << idx << " must be a StringAttr, got " << parameter;
    if (!declared.insert(name.getValue()).second)
      return emitOpError("duplicate parameter name '")
             << name.getValue() << "'";
  }

  if (!fnTypeAttr)
    return emitOpError(
        "parameters requires function_type to define the runtime SSA "
        "operand signature");

  llvm::SmallDenseSet<StringRef> keywordOnlyNames;
  if (ArrayAttr keywordOnly = getKeywordOnlyAttr()) {
    for (Attribute kw : keywordOnly) {
      auto kwName = dyn_cast<StringAttr>(kw);
      if (!kwName)
        return emitOpError("keyword_only entry must be a StringAttr, got ")
               << kw;
      StringRef name = kwName.getValue();
      if (!keywordOnlyNames.insert(name).second)
        return emitOpError("duplicate keyword_only entry '") << name << "'";
      if (!declared.contains(name))
        return emitOpError("keyword_only entry '")
               << name << "' is not listed in parameters";
    }
  }

  bool seenKeywordOnly = false;
  for (Attribute parameter : parameters) {
    StringRef name = cast<StringAttr>(parameter).getValue();
    if (keywordOnlyNames.contains(name)) {
      seenKeywordOnly = true;
      continue;
    }
    if (seenKeywordOnly)
      return emitOpError("positional parameter '")
             << name << "' cannot follow a keyword-only parameter";
  }

  if (ArrayAttr constKwargs = getConstKwargsAttr()) {
    llvm::SmallDenseSet<StringRef> seenConstKwargs;
    for (Attribute kw : constKwargs) {
      auto kwName = dyn_cast<StringAttr>(kw);
      if (!kwName)
        return emitOpError("const_kwargs entry must be a StringAttr, got ")
               << kw;
      StringRef name = kwName.getValue();
      if (!seenConstKwargs.insert(name).second)
        return emitOpError("duplicate const_kwargs entry '") << name << "'";
      if (!declared.contains(name))
        return emitOpError("const_kwargs entry '")
               << name << "' is not listed in parameters";
      if (!keywordOnlyNames.contains(name))
        return emitOpError("const_kwargs entry '")
               << name << "' must be listed in keyword_only";
    }
  }

  auto fnType = llvm::cast<FunctionType>(fnTypeAttr.getValue());
  ArrayAttr operandParameters =
      filterIntrinsicOperandParameters(parameters, getConstKwargsAttr());
  if (fnType.getNumInputs() != operandParameters.size())
    return emitOpError("function_type declares ")
           << fnType.getNumInputs()
           << " input(s) but non-const parameters declare "
           << operandParameters.size() << " runtime SSA operand(s)";
  return success();
}

//===----------------------------------------------------------------------===//
// `hc.return`.
//
// `hc.return` is not a required terminator (its callee-like parents carry
// `NoTerminator`), but when it appears it must be consistent with the
// enclosing callable's signature: kernels never return a value, and
// funcs/intrinsics with a declared `function_type` must return operands that
// match the declared result types.
//===----------------------------------------------------------------------===//

LogicalResult HCReturnOp::verify() {
  // Walk outward through control-flow / scope regions until we hit a
  // callable parent. `hc.subgroup_region`, `hc.workitem_region`,
  // `hc.for_range`, and `hc.if` are transparent to `hc.return`: the return
  // terminates the enclosing kernel/func/intrinsic, not the structured
  // region it textually sits in.
  Operation *callee = (*this)->getParentOp();
  while (callee && !isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(callee))
    callee = callee->getParentOp();
  if (!callee)
    return success();

  // Kernels never return values, irrespective of whether a signature was
  // declared. `HCKernelOp::verify` rejects result types in the signature;
  // this enforces the symmetric rule on the terminator side.
  if (isa<HCKernelOp>(callee)) {
    if (!getValues().empty())
      return emitOpError("`hc.return` inside `hc.kernel` must be operand-less; "
                         "kernels never produce a value");
    return success();
  }

  TypeAttr fnTypeAttr;
  if (auto f = dyn_cast<HCFuncOp>(callee))
    fnTypeAttr = f.getFunctionTypeAttr();
  else if (auto i = dyn_cast<HCIntrinsicOp>(callee))
    fnTypeAttr = i.getFunctionTypeAttr();
  if (!fnTypeAttr)
    return success();
  auto fnType = llvm::cast<FunctionType>(fnTypeAttr.getValue());
  if (getValues().size() != fnType.getNumResults())
    return emitOpError("returns ")
           << getValues().size() << " value(s) but enclosing "
           << callee->getName() << " declares " << fnType.getNumResults()
           << " result(s)";
  for (auto [i, returned, declared] :
       llvm::enumerate(getValues().getTypes(), fnType.getResults())) {
    if (!compatibleSigType(returned, declared))
      return emitOpError("returned value #")
             << i << " type " << returned << " does not match enclosing "
             << callee->getName() << " result type " << declared;
  }
  return success();
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

OpFoldResult HCConstOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

OpFoldResult HCUndefValueOp::fold(FoldAdaptor /*adaptor*/) {
  return UnitAttr::get(getContext());
}

OpFoldResult HCSymbolOp::fold(FoldAdaptor /*adaptor*/) {
  return TypeAttr::get(getResult().getType());
}

void HCYieldOp::getSuccessorRegions(ArrayRef<Attribute> /*operands*/,
                                    SmallVectorImpl<RegionSuccessor> &regions) {
  auto branch =
      dyn_cast_or_null<RegionBranchOpInterface>((*this)->getParentOp());
  if (!branch)
    return;
  branch.getSuccessorRegions(
      cast<RegionBranchTerminatorOpInterface>(getOperation()), regions);
}

OperandRange
HCForRangeOp::getEntrySuccessorOperands(RegionSuccessor /*successor*/) {
  return getIterInits();
}

void HCForRangeOp::getSuccessorRegions(
    RegionBranchPoint /*point*/, SmallVectorImpl<RegionSuccessor> &regions) {
  // Bounds are symbolic in HC, so the dialect cannot decide whether the loop
  // executes. Model both zero-trip and body-entry/iteration edges.
  regions.push_back(RegionSuccessor(&getBody()));
  regions.push_back(RegionSuccessor::parent());
}

ValueRange HCForRangeOp::getSuccessorInputs(RegionSuccessor successor) {
  if (successor.isParent())
    return getResults();
  if (getBody().empty())
    return {};
  return getBody().front().getArguments().drop_front();
}

bool HCForRangeOp::areTypesCompatible(Type lhs, Type rhs) {
  return compatibleBranchType(lhs, rhs);
}

void HCIfOp::getSuccessorRegions(RegionBranchPoint point,
                                 SmallVectorImpl<RegionSuccessor> &regions) {
  if (!point.isParent()) {
    regions.push_back(RegionSuccessor::parent());
    return;
  }

  regions.push_back(RegionSuccessor(&getThenRegion()));
  if (getElseRegion().empty())
    regions.push_back(RegionSuccessor::parent());
  else
    regions.push_back(RegionSuccessor(&getElseRegion()));
}

ValueRange HCIfOp::getSuccessorInputs(RegionSuccessor successor) {
  return successor.isParent() ? ValueRange(getResults()) : ValueRange();
}

bool HCIfOp::areTypesCompatible(Type lhs, Type rhs) {
  return compatibleBranchType(lhs, rhs);
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
    if (!compatibleBranchType(init.getType(), result.getType()))
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
    if (compatibleBranchType(initTy, blockTy))
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
    if (compatibleBranchType(yieldedTy, resultTy))
      continue;
    return emitOpError("body yield[")
           << idx << "] type " << yieldedTy << " does not match result[" << idx
           << "] type " << resultTy;
  }
  return success();
}

// Shared verifier for `hc.workitem_region` / `hc.subgroup_region`.
//
// Two legal body shapes:
//   1. `$results` empty — pre-promotion, side-effect-only, or
//      `hc.return` fall-through. We don't care what the terminator is
//      (or whether one exists at all: `NoTerminator` is the trait).
//   2. `$results` non-empty — post-promotion. Body must end with
//      `hc.yield`, arity matches `$results`, each value's type is
//      compatible with the corresponding result type (`!hc.undef`
//      escape applies on either side, matching progressive typing). Unlike
//      `hc.for_range` / `hc.if`, nested scopes are not region-branch ops and do
//      not widen distinct symbolic payloads across control-flow edges.
//
// A `hc.region_return` terminator combined with non-empty `$results`
// is the frontend contradicting itself — "pre-promotion" (the
// terminator) and "post-promotion" (declared results) simultaneously.
// That falls out of the rule above: path 2 requires `hc.yield`, so
// `hc.region_return` there is rejected.
static LogicalResult verifyNestedScopeRegion(Operation *op) {
  if (op->getNumResults() == 0)
    return success();
  Region &body = op->getRegion(0);
  if (body.empty() || body.front().empty())
    return op->emitOpError("declares ")
           << op->getNumResults() << " result(s) but body is empty";
  Operation *term = &body.front().back();
  auto yield = dyn_cast<HCYieldOp>(term);
  if (!yield)
    return op->emitOpError("declares results; body must terminate with "
                           "`hc.yield`, got ")
           << term->getName();
  if (yield.getValues().size() != op->getNumResults())
    return op->emitOpError("body yield produces ")
           << yield.getValues().size() << " values, expected "
           << op->getNumResults();
  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip_equal(yield.getValues(), op->getResults()))) {
    auto [yielded, result] = pair;
    Type yieldedTy = yielded.getType();
    Type resultTy = result.getType();
    if (yieldedTy == resultTy || isUndef(yieldedTy) || isUndef(resultTy))
      continue;
    return op->emitOpError("body yield[")
           << idx << "] type " << yieldedTy << " does not match result[" << idx
           << "] type " << resultTy;
  }
  return success();
}

LogicalResult HCWorkitemRegionOp::verify() {
  return verifyNestedScopeRegion(*this);
}

LogicalResult HCSubgroupRegionOp::verify() {
  return verifyNestedScopeRegion(*this);
}

// Each `$names` entry stands for one result + one writeback assign
// once `-hc-promote-names` rebuilds the parent region. Two entries
// naming the same slot would produce two results for the same name
// and two same-named writebacks in the enclosing store — ambiguous
// on the write side, meaningless on the read side. The
// `StrArrayAttr` ODS constraint guarantees element types, but a
// defensive re-check keeps the diagnostic tied to this op if a
// future type relaxation ever changes that.
LogicalResult HCRegionReturnOp::verify() {
  llvm::SmallPtrSet<StringAttr, 4> seen;
  for (auto [idx, raw] : llvm::enumerate(getNames())) {
    auto name = llvm::dyn_cast<StringAttr>(raw);
    if (!name)
      return emitOpError("`names[")
             << idx << "]` is not a StringAttr (got " << raw << ")";
    if (!seen.insert(name).second)
      return emitOpError("duplicate name '")
             << name.getValue()
             << "' in `names`; each entry surfaces as a distinct result and "
             << "spawns one writeback assign — duplicates would alias on "
             << "both sides";
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
      if (compatibleBranchType(yieldedTy, resultTy))
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

/// Extract a concrete shape attribute from a shaped `hc` type, or `nullptr`
/// if the type does not (yet) carry one. Pre-inference IR is typically
/// `!hc.undef`, in which case rank is unknown and axis range cannot be
/// checked — later inference refines the type and picks up the check.
///
/// Fully-qualified `mlir::hc::{Buffer,Tensor,Vector}Type` are required to
/// avoid colliding with MLIR's builtin `VectorType`/`TensorType`.
static ShapeAttr tryGetShape(Type t) {
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
static Operation *findNarrowingScope(Operation *op) {
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
static LogicalResult verifyTensorAllocScope(Operation *op) {
  if (Operation *narrowing = findNarrowingScope(op))
    return op->emitOpError("tensor allocator is workgroup scope only; "
                           "enclosed by ")
           << narrowing->getName() << " which narrows the scope";
  return success();
}

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

/// Verify that an optional `#hc.shape` on a load/vload op matches the
/// number of index operands one-to-one. Without a shape attr, we cannot
/// check rank — leave that to inference-stage checks.
template <typename OpT>
static LogicalResult
verifyLoadShapeRank(OpT op, mlir::Operation::operand_range indices) {
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

LogicalResult HCLoadOp::verify() {
  return verifyLoadShapeRank(*this, getIndices());
}

LogicalResult HCVLoadOp::verify() {
  return verifyLoadShapeRank(*this, getIndices());
}

LogicalResult HCGetItemOp::verify() {
  if (getIndices().empty())
    return emitOpError("expected at least one index");
  if (isa<TupleType>(getBase().getType()) && getIndices().size() != 1)
    return emitOpError(
               "expected exactly one index when the base is a tuple, got ")
           << getIndices().size();
  return success();
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

template <typename CallOp>
static LogicalResult verifySignature(CallOp op, FunctionType fnType) {
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
    if (!compatibleBranchType(callSite, declared))
      return op.emitOpError("arg #")
             << i << " type " << callSite
             << " is incompatible with callee declaration " << declared;
  }
  for (auto [i, callSite, declared] :
       llvm::enumerate(op.getResults().getTypes(), fnType.getResults())) {
    if (!compatibleBranchType(callSite, declared))
      return op.emitOpError("result #")
             << i << " type " << callSite
             << " is incompatible with callee declaration " << declared;
  }
  return success();
}

template <typename CalleeOp, typename CallOp>
static LogicalResult
verifyFlatSymbolUseAsOp(CallOp op, SymbolTableCollection &symbolTable,
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

LogicalResult HCCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  return verifyFlatSymbolUseAsOp<HCFuncOp>(*this, symbolTable, "hc.func");
}

CallInterfaceCallable HCCallOp::getCallableForCallee() {
  return getCalleeAttr();
}

void HCCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  auto symbol = cast<SymbolRefAttr>(callee);
  (*this)->setAttr(getCalleeAttrName(), cast<FlatSymbolRefAttr>(symbol));
}

Operation::operand_range HCCallOp::getArgOperands() { return getArgs(); }

MutableOperandRange HCCallOp::getArgOperandsMutable() {
  return getArgsMutable();
}

// Translates the declared effect class into concrete side effects on the
// default resource. The callee's body is opaque; we only know "maybe reads"
// / "maybe writes" at this level, so `Pure` emits nothing, the one-sided
// classes emit the matching effect, and the unknown/absent case falls back
// to MemRead+MemWrite.
static void emitEffectsForClass(
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
static void populateEffectsFromCallee(
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
