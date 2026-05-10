// SPDX-FileCopyrightText: 2026 hc contributors
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "hc/IR/HCOps.h"

#include "hc/IR/HCDialect.h"
#include "hc/IR/HCSymbols.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringSet.h"

#include <optional>

using namespace mlir;
using namespace mlir::hc;

#include "hc/IR/HCOpsInterfaces.cpp.inc"

// Forward declarations for `custom<...>(...)` directives consumed by the
// tablegen-generated op parse/print methods. The definitions live below
// the generated include so the helpers can use the generated op classes.
static mlir::ParseResult parseHCAsLayoutAttr(mlir::OpAsmParser &parser,
                                             mlir::Attribute &layout);
static void printHCAsLayoutAttr(mlir::OpAsmPrinter &printer,
                                mlir::Operation *op, mlir::Attribute layout);

#define GET_OP_CLASSES
#include "hc/IR/HCOps.cpp.inc"

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

// Custom parse/print for `hc.as_layout`'s `$layout` operand. The slot
// admits both the legacy `row_major` / `col_major` keyword (an
// `HC_NamedLayoutAttr` enum) and the structured `#hc.layout<...>`
// descriptor (an `HC_LayoutAttr`). The keyword form is what every
// existing IR uses; the structured form is what `doc/layouts.md`
// commits to. Picking which one to emit is a syntactic decision, not a
// semantic one — both lower into the same `$layout` slot — so the
// dispatch lives here, not in the verifier.
static ParseResult parseHCAsLayoutAttr(OpAsmParser &parser, Attribute &layout) {
  StringRef keyword;
  // `row_major` / `col_major` are keywords — peek ahead for a bare
  // identifier first. If the next token is `#` (the dialect-prefixed
  // attribute lead-in) `parseOptionalKeyword` declines and we fall
  // through to the structured-attribute parser.
  if (succeeded(parser.parseOptionalKeyword(&keyword))) {
    if (auto value = symbolizeNamedLayout(keyword)) {
      layout = NamedLayoutAttr::get(parser.getContext(), *value);
      return success();
    }
    return parser.emitError(parser.getCurrentLocation())
           << "expected `row_major`, `col_major`, or `(#hc.layout<...>)`, "
              "got '"
           << keyword << "'";
  }
  // Structured form is wrapped in `(...)` because MLIR's
  // `parseExtendedAttr` unconditionally consumes a trailing `: type`
  // after a dialect-prefixed attribute (the type annotation for typed
  // attrs); without the parens it would eat the assembly format's
  // literal `:` separator that precedes `type($value)`. The parens
  // ensure the lookahead sees `)` instead, leaving the literal `:` for
  // the format to consume.
  LayoutAttr structured;
  if (parser.parseLParen() || parser.parseAttribute(structured) ||
      parser.parseRParen())
    return failure();
  layout = structured;
  return success();
}

static void printHCAsLayoutAttr(OpAsmPrinter &printer, Operation *op,
                                Attribute layout) {
  // Round-trip the keyword form when we can — it is what every existing
  // surface IR uses and what `verify-hc.mlir` round-trips against. The
  // structured form is wrapped in parens to mirror the parser; see
  // `parseHCAsLayoutAttr` for the rationale.
  if (auto named = llvm::dyn_cast<NamedLayoutAttr>(layout)) {
    printer << stringifyNamedLayout(named.getValue());
    return;
  }
  printer << "(";
  printer.printAttribute(layout);
  printer << ")";
}

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
  if (auto boundSymbols = getBoundSymbolsAttr()) {
    llvm::StringSet<> seen;
    for (StringAttr symbol : boundSymbols.getAsRange<StringAttr>()) {
      StringRef name = symbol.getValue();
      if (name.empty())
        return emitOpError("bound symbol names must be non-empty");
      if (!seen.insert(name).second)
        return emitOpError("duplicate bound symbol '") << name << "'";
    }
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
    if (!areHCProgressiveTypesCompatible(returned, declared))
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

LogicalResult HCMaterializeBoundExprOp::verify() {
  Type result = getResult().getType();
  if (auto idx = llvm::dyn_cast<IdxType>(result)) {
    if (idx.getExpr())
      return success();
    return emitOpError("result must pin a bound symbolic expression");
  }
  if (auto pred = llvm::dyn_cast<PredType>(result)) {
    if (pred.getPred())
      return success();
    return emitOpError("result must pin a bound symbolic predicate");
  }
  return emitOpError("result must be a pinned `!hc.idx` or `!hc.pred`, got ")
         << result;
}

OpFoldResult HCConstOp::fold(FoldAdaptor /*adaptor*/) { return getValue(); }

OpFoldResult HCUndefValueOp::fold(FoldAdaptor /*adaptor*/) {
  return UnitAttr::get(getContext());
}

OpFoldResult HCSymbolOp::fold(FoldAdaptor /*adaptor*/) {
  return TypeAttr::get(getResult().getType());
}

struct LaunchContextRanks {
  std::optional<unsigned> workRank;
  std::optional<unsigned> groupRank;
};

static std::optional<unsigned> shapeRank(ShapeAttr shape) {
  if (!shape)
    return std::nullopt;
  return static_cast<unsigned>(shape.getDims().size());
}

static LaunchContextRanks launchContextRanks(Type type) {
  if (auto group = dyn_cast<GroupType>(type))
    return {shapeRank(group.getWorkShape()), shapeRank(group.getGroupShape())};
  if (auto workitem = dyn_cast<WorkitemType>(type))
    return {/*workRank=*/std::nullopt, shapeRank(workitem.getGroupShape())};
  if (auto subgroup = dyn_cast<SubgroupType>(type))
    return {/*workRank=*/std::nullopt, shapeRank(subgroup.getGroupShape())};
  return {};
}

static std::optional<unsigned> expectedLaunchGeometryResults(Operation *op,
                                                             Type contextType) {
  if (isa<HCGroupSizeOp, HCWaveSizeOp>(op))
    return 1u;

  LaunchContextRanks ranks = launchContextRanks(contextType);
  if (isa<HCGroupIdOp, HCWorkOffsetOp, HCWorkShapeOp>(op))
    return ranks.workRank;
  if (isa<HCLocalIdOp, HCSubgroupIdOp, HCGroupShapeOp>(op))
    return ranks.groupRank;
  return std::nullopt;
}

template <typename OpT> static LogicalResult verifyLaunchGeometryArity(OpT op) {
  std::optional<unsigned> expected =
      expectedLaunchGeometryResults(op.getOperation(), op.getGroup().getType());
  if (!expected || op->getNumResults() == *expected)
    return success();
  return op.emitOpError("expected ")
         << *expected << " result(s) for launch-geometry query, got "
         << op->getNumResults();
}

LogicalResult HCGroupIdOp::verify() { return verifyLaunchGeometryArity(*this); }

LogicalResult HCLocalIdOp::verify() { return verifyLaunchGeometryArity(*this); }

LogicalResult HCSubgroupIdOp::verify() {
  return verifyLaunchGeometryArity(*this);
}

LogicalResult HCGroupShapeOp::verify() {
  return verifyLaunchGeometryArity(*this);
}

LogicalResult HCGroupSizeOp::verify() {
  return verifyLaunchGeometryArity(*this);
}

LogicalResult HCWorkOffsetOp::verify() {
  return verifyLaunchGeometryArity(*this);
}

LogicalResult HCWorkShapeOp::verify() {
  return verifyLaunchGeometryArity(*this);
}

LogicalResult HCWaveSizeOp::verify() {
  return verifyLaunchGeometryArity(*this);
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
  return areHCBranchTypesCompatible(lhs, rhs);
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
  return areHCBranchTypesCompatible(lhs, rhs);
}

template <typename RegionOpT>
static ValueRange getSingleBlockYieldedResultValues(RegionOpT op) {
  if (op->getNumResults() == 0)
    return {};
  Region &body = op.getBody();
  if (body.empty() || body.front().empty())
    return {};
  auto yield = dyn_cast<HCYieldOp>(body.front().getTerminator());
  if (!yield)
    return {};
  return yield.getValues();
}

ValueRange HCSubgroupRegionOp::getYieldedResultValues() {
  return getSingleBlockYieldedResultValues(*this);
}

ValueRange HCWorkitemRegionOp::getYieldedResultValues() {
  return getSingleBlockYieldedResultValues(*this);
}

static FailureOr<ExprAttr> composeCollectiveDim(Operation *op, ExprAttr lhs,
                                                sym::ExprBinaryOp opKind,
                                                ExprAttr rhs) {
  MLIRContext *ctx = op->getContext();
  std::string diag;
  FailureOr<sym::ExprHandle> handle = sym::composeExprBinary(
      ctx->getOrLoadDialect<HCDialect>()->getSymbolStore(),
      sym::ExprHandle(lhs.getNode()), opKind, sym::ExprHandle(rhs.getNode()),
      &diag);
  if (failed(handle))
    return failure();
  return ExprAttr::get(ctx, *handle);
}

static FailureOr<SmallVector<Attribute>>
subgroupCollectiveSuffix(Operation *op, LaunchContextMetadata metadata) {
  if (!metadata.groupShape || !metadata.subgroupSize ||
      metadata.groupShape.getDims().empty())
    return SmallVector<Attribute>{};

  auto firstDim = dyn_cast<ExprAttr>(metadata.groupShape.getDims().front());
  if (!firstDim)
    return SmallVector<Attribute>{};

  FailureOr<ExprAttr> count = composeCollectiveDim(
      op, firstDim, sym::ExprBinaryOp::Div, metadata.subgroupSize);
  if (failed(count))
    return failure();

  for (Attribute dim : metadata.groupShape.getDims().drop_front()) {
    auto expr = dyn_cast<ExprAttr>(dim);
    if (!expr)
      return SmallVector<Attribute>{};
    count = composeCollectiveDim(op, *count, sym::ExprBinaryOp::Mul, expr);
    if (failed(count))
      return failure();
  }
  return SmallVector<Attribute>{*count};
}

static FailureOr<SmallVector<Attribute>>
collectiveSuffix(Operation *op, LaunchContextMetadata metadata) {
  if (isa<HCWorkitemRegionOp>(op)) {
    if (!metadata.groupShape)
      return SmallVector<Attribute>{};
    return SmallVector<Attribute>(metadata.groupShape.getDims());
  }
  if (isa<HCSubgroupRegionOp>(op))
    return subgroupCollectiveSuffix(op, metadata);
  return SmallVector<Attribute>{};
}

static bool isCollectiveScalarType(Type type) {
  return isa<IdxType, PredType>(type) || type.isIntOrIndexOrFloat();
}

static Type appendCollectiveSuffixToVector(Type type,
                                           ArrayRef<Attribute> suffix) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
  if (!shaped)
    return {};
  SmallVector<Attribute> dims(shaped.getSymbolicShape().getDims());
  llvm::append_range(dims, suffix);
  ShapeAttr shape = ShapeAttr::get(type.getContext(), dims);
  if (auto vector = dyn_cast<mlir::hc::VectorType>(type))
    return mlir::hc::VectorType::get(type.getContext(), vector.getElementType(),
                                     shape);
  if (auto bareVector = dyn_cast<mlir::hc::BareVectorType>(type))
    return mlir::hc::BareVectorType::get(type.getContext(),
                                         bareVector.getElementType(), shape);
  return {};
}

static Type collectiveLiftedType(Type yieldedType, ArrayRef<Attribute> suffix) {
  if (suffix.empty() || !yieldedType || isHCUndefType(yieldedType))
    return yieldedType;

  if (auto tuple = dyn_cast<TupleType>(yieldedType)) {
    SmallVector<Type> elements;
    elements.reserve(tuple.size());
    for (Type element : tuple.getTypes()) {
      if (isa<TupleType>(element))
        return {};
      Type lifted = collectiveLiftedType(element, suffix);
      if (!lifted)
        return {};
      elements.push_back(lifted);
    }
    return TupleType::get(yieldedType.getContext(), elements);
  }

  if (Type vector = appendCollectiveSuffixToVector(yieldedType, suffix))
    return vector;

  if (isCollectiveScalarType(yieldedType)) {
    ShapeAttr shape = ShapeAttr::get(yieldedType.getContext(), suffix);
    return mlir::hc::VectorType::get(yieldedType.getContext(), yieldedType,
                                     shape);
  }
  return {};
}

static bool collectiveYieldMatchesRegionResult(Operation *op, Type yieldedType,
                                               Type resultType) {
  if (!isa<HCWorkitemRegionOp, HCSubgroupRegionOp>(op) ||
      op->getRegion(0).empty() ||
      op->getRegion(0).front().getNumArguments() == 0)
    return false;
  std::optional<LaunchContextMetadata> metadata = getLaunchContextMetadata(
      op->getRegion(0).front().getArgument(0).getType());
  if (!metadata)
    return false;
  FailureOr<SmallVector<Attribute>> suffix = collectiveSuffix(op, *metadata);
  if (failed(suffix))
    return false;
  return collectiveLiftedType(yieldedType, *suffix) == resultType;
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
    if (!areHCBranchTypesCompatible(init.getType(), result.getType()))
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
    if (areHCBranchTypesCompatible(initTy, blockTy))
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
    if (areHCBranchTypesCompatible(yieldedTy, resultTy))
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
//      escape applies on either side, matching progressive typing).
//      Collective region results additionally accept the source-level lifting
//      rule where each yielded scalar/vector gains the region's participant
//      suffix in the enclosing scope.
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
    if (areHCBranchTypesCompatible(yieldedTy, resultTy))
      continue;
    if (collectiveYieldMatchesRegionResult(op, yieldedTy, resultTy))
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
      if (areHCBranchTypesCompatible(yieldedTy, resultTy))
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

/// Extract a concrete shape attribute from a symbolically shaped `hc` type, or
/// `nullptr` if the type does not (yet) carry one. Pre-inference IR is
/// typically `!hc.undef`, in which case rank is unknown and axis range cannot
/// be checked — later inference refines the type and picks up the check.
static ShapeAttr tryGetShape(Type t) {
  if (auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(t))
    return shaped.getSymbolicShape();
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

LogicalResult HCTupleOp::verify() {
  Type resultType = getResult().getType();
  if (isHCUndefType(resultType))
    return success();
  auto tuple = dyn_cast<TupleType>(resultType);
  if (!tuple)
    return emitOpError("result must be !hc.undef or builtin tuple, got ")
           << resultType;
  if (tuple.size() != getElements().size())
    return emitOpError("result tuple arity ")
           << tuple.size() << " does not match element count "
           << getElements().size();
  for (auto [index, pair] :
       llvm::enumerate(llvm::zip_equal(getElements(), tuple.getTypes()))) {
    auto [value, type] = pair;
    if (areHCBranchTypesCompatible(value.getType(), type))
      continue;
    return emitOpError("element #")
           << index << " type " << value.getType()
           << " does not match result tuple element type " << type;
  }
  return success();
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
  if (isHCUndefType(result))
    return success();
  if (result.isIntOrIndexOrFloat()) {
    if (result != target)
      return emitOpError("result type ")
             << result << " does not match target type " << target;
    return success();
  }
  auto elementOf = [](Type t) -> Type {
    if (!llvm::isa<mlir::hc::TensorType, mlir::hc::VectorType>(t))
      return {};
    if (auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(t))
      return shaped.getSymbolicElementType();
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
  // `$inactive` is a scalar SSA value; once inference gives both operands
  // meaningful domains, it must agree with the masked value's element type.
  Type value = getValue().getType();
  if (isHCUndefType(value))
    return success();
  Type elem;
  if (auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(value))
    elem = shaped.getSymbolicElementType();
  if (!elem || isHCUndefType(elem))
    return success();
  Type inactive = getInactive().getType();
  if (isHCUndefType(inactive))
    return success();
  auto sameDomain = [&](Type t) -> bool {
    if (llvm::isa<PredType>(t))
      return elem.isInteger(1);
    if (llvm::isa<IdxType>(t))
      return elem.isIntOrIndex();
    if (t.isIntOrIndexOrFloat())
      return elem == t;
    return false;
  };
  if (!sameDomain(inactive))
    return emitOpError("inactive value type ")
           << inactive << " does not match element type " << elem;
  return success();
}

static Type bareShapedElement(Type type) {
  if (!llvm::isa<BareTensorType, BareVectorType>(type))
    return {};
  auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(type);
  return shaped ? shaped.getSymbolicElementType() : Type{};
}

static Type barePredicateMaskType(Type type) {
  auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(type);
  if (!shaped)
    return {};
  Type pred = getUnpinnedPredType(type.getContext());
  if (llvm::isa<BareTensorType>(type))
    return BareTensorType::get(type.getContext(), pred,
                               shaped.getSymbolicShape());
  if (llvm::isa<BareVectorType>(type))
    return BareVectorType::get(type.getContext(), pred,
                               shaped.getSymbolicShape());
  return {};
}

static LogicalResult verifyBarePredicateMask(Operation *op, Type type) {
  Type elem = bareShapedElement(type);
  if (!elem || !llvm::isa<PredType>(elem))
    return op->emitOpError("result type must be a bare tensor/vector of "
                           "`!hc.pred`, got ")
           << type;
  return success();
}

LogicalResult HCLoadMaskOp::verify() {
  return verifyBarePredicateMask(getOperation(), getMask().getType());
}

LogicalResult HCFullMaskOp::verify() {
  return verifyBarePredicateMask(getOperation(), getMask().getType());
}

LogicalResult HCStoreOp::verify() {
  Value mask = getMask();
  Type source = getSource().getType();

  if (!mask)
    return success();

  Type expectedMask = barePredicateMaskType(source);
  if (!expectedMask)
    return emitOpError(
               "mask operand requires a bare tensor/vector source, got ")
           << source;
  if (mask.getType() != expectedMask)
    return emitOpError("mask type ")
           << mask.getType() << " must match source validity type "
           << expectedMask;
  return verifyBarePredicateMask(getOperation(), mask.getType());
}

LogicalResult HCSelectOp::verify() {
  Type condition = getCondition().getType();
  if (failed(verifyBarePredicateMask(getOperation(), condition)))
    return failure();

  Type trueValue = getTrueValue().getType();
  Type result = getResult().getType();
  if (trueValue != result)
    return emitOpError("true value type ")
           << trueValue << " must match result type " << result;

  Type elem = bareShapedElement(result);
  Type inactive = getFalseValue().getType();
  if (!elem || isHCUndefType(inactive))
    return success();
  auto sameDomain = [&](Type t) -> bool {
    if (llvm::isa<PredType>(t))
      return elem.isInteger(1);
    if (llvm::isa<IdxType>(t))
      return elem.isIntOrIndex();
    if (t.isIntOrIndexOrFloat())
      return elem == t;
    return false;
  };
  if (!sameDomain(inactive))
    return emitOpError("false value type ")
           << inactive << " does not match result element type " << elem;
  return success();
}

Value HCLoadOp::getStaticShapeOperand() { return getShape(); }
Type HCLoadOp::getStaticShapedResultType() { return getResult().getType(); }
bool HCLoadOp::hasStaticVectorResult() { return false; }
Value HCLoadOp::getStaticShapeSourceOperand() { return getBuffer(); }
ValueRange HCLoadOp::getStaticShapeIndexOperands() { return getIndices(); }

Value HCVLoadOp::getStaticShapeOperand() { return getShape(); }
Type HCVLoadOp::getStaticShapedResultType() { return getResult().getType(); }
bool HCVLoadOp::hasStaticVectorResult() { return true; }
Value HCVLoadOp::getStaticShapeSourceOperand() { return getSource(); }
ValueRange HCVLoadOp::getStaticShapeIndexOperands() { return getIndices(); }

LogicalResult HCZerosOp::verify() { return verifyTensorAllocScope(*this); }
LogicalResult HCOnesOp::verify() { return verifyTensorAllocScope(*this); }
LogicalResult HCFullOp::verify() { return verifyTensorAllocScope(*this); }
LogicalResult HCEmptyOp::verify() { return verifyTensorAllocScope(*this); }

//===----------------------------------------------------------------------===//
// hc.ptr_offset / hc.ptr_load / hc.ptr_store verifiers.
//
// All three share the rule that the pointer's `addrspace` and (when
// present) `elementType` must agree with the matching operand/result.
// `hc.ptr_load` / `hc.ptr_store` accept scalar or `vector<NxT>` value
// types — for vectors the parity check runs against the vector's
// element type, the width is informational and stays unconstrained
// here (the LLVM-lowering boundary owns hardware-width splitting).
// Pre-inference IR with `!hc.undef` on either side escapes the parity
// check, mirroring how the rest of the dialect tolerates the
// progressive-typing placeholder.
//===----------------------------------------------------------------------===//

namespace {

// Returns the `PtrType` payload, or null when the operand is still
// `!hc.undef`. Anything else is an unreachable verifier-time bug because
// `HC_PtrValueType` already restricted the constraint.
static PtrType ptrTypeOrUndef(Type type) {
  if (isHCUndefType(type))
    return {};
  return llvm::cast<PtrType>(type);
}

// Element-type compatibility for ptr_load / ptr_store: typed pointers
// require an exact match against the value's element type, opaque
// pointers (no element type on the pointer) accept anything. The value
// may be a scalar (its own type is the element type) or an upstream
// `vector<NxT>` (the vector's element type is the element type — the
// vector denotes a contiguous N-element access starting at the
// pointer). `!hc.undef` on either side escapes — the caller has
// already filtered out the `!hc.ptr` shell.
static LogicalResult checkLoadStoreValueMatchesPointer(Operation *op,
                                                       Type valueType,
                                                       PtrType ptr,
                                                       StringRef role) {
  if (!ptr || isHCUndefType(valueType))
    return success();
  Type ptrElem = ptr.getElementType();
  if (!ptrElem)
    return success();
  Type elementType = valueType;
  if (auto vec = llvm::dyn_cast<mlir::VectorType>(valueType))
    elementType = vec.getElementType();
  if (elementType == ptrElem)
    return success();
  return op->emitOpError(role)
         << " element type " << elementType
         << " must match pointer element type " << ptrElem;
}

} // namespace

LogicalResult HCPtrOffsetOp::verify() {
  PtrType source = ptrTypeOrUndef(getSource().getType());
  PtrType result = ptrTypeOrUndef(getResult().getType());
  if (!source || !result)
    return success();
  if (source.getAddrSpace() != result.getAddrSpace())
    return emitOpError("address space mismatch: source ")
           << stringifyAddrSpace(source.getAddrSpace()) << " vs result "
           << stringifyAddrSpace(result.getAddrSpace());
  Type sourceElem = source.getElementType();
  Type resultElem = result.getElementType();
  if (sourceElem && resultElem && sourceElem != resultElem)
    return emitOpError("element type mismatch: source ")
           << sourceElem << " vs result " << resultElem;
  // Asymmetric element-type loss (typed source, opaque result, or vice
  // versa) is rejected: the dropping happens at the LLVM-lowering
  // boundary in one go, not piecemeal at every offset.
  if (static_cast<bool>(sourceElem) != static_cast<bool>(resultElem))
    return emitOpError(
        "source and result must agree on whether the pointer is typed");
  return success();
}

LogicalResult HCPtrLoadOp::verify() {
  PtrType source = ptrTypeOrUndef(getSource().getType());
  return checkLoadStoreValueMatchesPointer(
      getOperation(), getResult().getType(), source, "result");
}

LogicalResult HCPtrStoreOp::verify() {
  PtrType dest = ptrTypeOrUndef(getDest().getType());
  return checkLoadStoreValueMatchesPointer(getOperation(), getValue().getType(),
                                           dest, "value");
}

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
    if (!areHCBranchTypesCompatible(callSite, declared))
      return op.emitOpError("arg #")
             << i << " type " << callSite
             << " is incompatible with callee declaration " << declared;
  }
  for (auto [i, callSite, declared] :
       llvm::enumerate(op.getResults().getTypes(), fnType.getResults())) {
    if (!areHCBranchTypesCompatible(callSite, declared))
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

//===----------------------------------------------------------------------===//
// hc.generic parse/print/verify.
//
// Custom assembly format because the iter / ins / outs clauses interleave
// SSA values, attributes, and types in a way the declarative format can't
// express directly. The shape is:
//
//   hc.generic
//       iter (parallel i = %m : index, reduction k = %kn : index)
//       ins  (%a at #hc.expr<"i*K+k"> : !hc.bare_tensor<...>, ...)
//       outs (%c at #hc.expr<"i*N+j"> : !hc.bare_tensor<...>)
//       -> (!hc.bare_tensor<...>, ...)
//       attributes { ... } {
//     ^bb0(%av: f16, %bv: f16, %cv: f32):
//       ...
//       hc.yield %s : f32
//   }
//
// `iter (...)` requires at least one entry; `outs (...)` requires at least
// one output. `ins (...)` may be empty.
//===----------------------------------------------------------------------===//

namespace {

// Symbolic-element extraction shared between the body-arg and yield checks.
// Returns the body-arg "element" the operand contributes:
//   - shaped HC types (incl. `!hc.buffer`)         -> shape's element type;
//   - `!hc.ptr` with an explicit pointee           -> pointee type;
//   - `!hc.ptr` opaque or `!hc.undef`              -> null escape.
// Null is the "no parity check" sentinel for progressive typing and for
// opaque pointers whose body type is decided at lowering time.
static Type genericOperandElement(Type type) {
  if (isHCUndefType(type))
    return {};
  if (auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(type))
    return shaped.getSymbolicElementType();
  if (auto ptr = llvm::dyn_cast<PtrType>(type))
    return ptr.getElementType();
  return {};
}

// Memory carriers (ptr / buffer) drive the per-operand effects on
// `hc.generic` and bypass the SSA-result slot — value-semantic outs
// produce results in declaration order, ptr/buffer outs do not.
// `!hc.undef` is conservatively treated as value-typed: the frontend
// emits undef before inference and pairs each with an explicit result;
// classifying it as ptr-like would silently drop that result.
static bool isMemoryCarrierOperand(Type type) {
  return llvm::isa<PtrType, BufferType>(type);
}

static ParseResult parseGenericIterClause(
    OpAsmParser &parser, SmallVectorImpl<Attribute> &iterSyms,
    SmallVectorImpl<Attribute> &iterKinds,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &boundsOps,
    SmallVectorImpl<Type> &boundsTypes) {
  if (parser.parseKeyword("iter") || parser.parseLParen())
    return failure();
  // Empty `iter ()` is rejected at parse — the verifier would catch it
  // anyway, but failing here gives a localized diagnostic.
  MLIRContext *ctx = parser.getContext();
  auto parseOne = [&]() -> ParseResult {
    StringRef kindKw;
    auto kindLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&kindKw))
      return failure();
    auto kind = symbolizeIterKind(kindKw);
    if (!kind)
      return parser.emitError(kindLoc)
             << "expected `parallel` or `reduction`, got '" << kindKw << "'";
    iterKinds.push_back(IterKindAttr::get(ctx, *kind));
    StringRef name;
    auto nameLoc = parser.getCurrentLocation();
    if (parser.parseKeyword(&name))
      return failure();
    if (name.empty())
      return parser.emitError(nameLoc) << "iter sym name must be non-empty";
    iterSyms.push_back(StringAttr::get(ctx, name));
    OpAsmParser::UnresolvedOperand bound;
    Type ty;
    if (parser.parseEqual() || parser.parseOperand(bound) ||
        parser.parseColonType(ty))
      return failure();
    boundsOps.push_back(bound);
    boundsTypes.push_back(ty);
    return success();
  };
  if (parseOne())
    return failure();
  while (succeeded(parser.parseOptionalComma()))
    if (parseOne())
      return failure();
  return parser.parseRParen();
}

static ParseResult parseGenericOperandClause(
    OpAsmParser &parser, StringRef keyword, bool allowEmpty,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ops,
    SmallVectorImpl<Type> &types, SmallVectorImpl<Attribute> &offsets) {
  if (parser.parseKeyword(keyword) || parser.parseLParen())
    return failure();
  if (succeeded(parser.parseOptionalRParen())) {
    if (!allowEmpty)
      return parser.emitError(parser.getCurrentLocation())
             << keyword << " clause requires at least one entry";
    return success();
  }
  MLIRContext *ctx = parser.getContext();
  auto parseOne = [&]() -> ParseResult {
    OpAsmParser::UnresolvedOperand op;
    Type ty;
    // Per-axis offsets ride inside `[...]`. The bracket terminator
    // shields the trailing `: type` from `parseExtendedAttr`, which
    // would otherwise consume the colon as part of the attribute. Empty
    // arrays are permitted at parse — the verifier flags rank mismatch
    // with a clearer message than a parser-level "missing entry" would.
    SmallVector<Attribute> axisOffsets;
    auto parseAxis = [&]() -> ParseResult {
      Attribute axisAttr;
      auto attrLoc = parser.getCurrentLocation();
      if (parser.parseAttribute(axisAttr))
        return failure();
      auto expr = llvm::dyn_cast<ExprAttr>(axisAttr);
      if (!expr)
        return parser.emitError(attrLoc) << "expected #hc.expr<...> attribute";
      axisOffsets.push_back(expr);
      return success();
    };
    if (parser.parseOperand(op) || parser.parseKeyword("at") ||
        parser.parseLSquare())
      return failure();
    if (failed(parser.parseOptionalRSquare())) {
      if (parseAxis())
        return failure();
      while (succeeded(parser.parseOptionalComma()))
        if (parseAxis())
          return failure();
      if (parser.parseRSquare())
        return failure();
    }
    if (parser.parseColonType(ty))
      return failure();
    ops.push_back(op);
    offsets.push_back(ArrayAttr::get(ctx, axisOffsets));
    types.push_back(ty);
    return success();
  };
  if (parseOne())
    return failure();
  while (succeeded(parser.parseOptionalComma()))
    if (parseOne())
      return failure();
  return parser.parseRParen();
}

static void printGenericOperandClause(OpAsmPrinter &p, StringRef keyword,
                                      OperandRange operands,
                                      ArrayAttr offsets) {
  p << ' ' << keyword << " (";
  llvm::interleaveComma(
      llvm::zip_equal(operands, offsets.getAsRange<ArrayAttr>()), p,
      [&](auto pair) {
        auto [val, perOperand] = pair;
        p << val << " at [";
        // `perOperand` is structured-bound from a generic-lambda parameter,
        // so the compiler treats it as dependent and needs `.template` to
        // resolve the member template.
        llvm::interleaveComma(perOperand.template getAsRange<ExprAttr>(), p,
                              [&](ExprAttr e) { p.printAttribute(e); });
        p << "] : " << val.getType();
      });
  p << ")";
}

} // namespace

ParseResult HCGenericOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<Attribute> iterSyms;
  SmallVector<Attribute> iterKinds;
  SmallVector<OpAsmParser::UnresolvedOperand> boundsOps;
  SmallVector<Type> boundsTypes;
  if (parseGenericIterClause(parser, iterSyms, iterKinds, boundsOps,
                             boundsTypes))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> insOps;
  SmallVector<Type> insTypes;
  SmallVector<Attribute> insOffsets;
  if (parseGenericOperandClause(parser, "ins", /*allowEmpty=*/true, insOps,
                                insTypes, insOffsets))
    return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> outsOps;
  SmallVector<Type> outsTypes;
  SmallVector<Attribute> outsOffsets;
  if (parseGenericOperandClause(parser, "outs", /*allowEmpty=*/false, outsOps,
                                outsTypes, outsOffsets))
    return failure();

  SmallVector<Type> resultTypes;
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseTypeList(resultTypes) || parser.parseRParen())
      return failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{},
                         /*enableNameShadowing=*/false))
    return failure();

  // `resolveOperands` appends to `result.operands` in the call order; the
  // ODS-declared operand groups (iter_bounds, ins, outs) must arrive in the
  // same order so `operandSegmentSizes` reads them back consistently.
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperands(boundsOps, boundsTypes, loc, result.operands) ||
      parser.resolveOperands(insOps, insTypes, loc, result.operands) ||
      parser.resolveOperands(outsOps, outsTypes, loc, result.operands))
    return failure();

  MLIRContext *ctx = parser.getContext();
  auto &builder = parser.getBuilder();
  result.addAttribute(getIterSymsAttrName(result.name),
                      ArrayAttr::get(ctx, iterSyms));
  result.addAttribute(getIterKindsAttrName(result.name),
                      ArrayAttr::get(ctx, iterKinds));
  result.addAttribute(getInsOffsetsAttrName(result.name),
                      ArrayAttr::get(ctx, insOffsets));
  result.addAttribute(getOutsOffsetsAttrName(result.name),
                      ArrayAttr::get(ctx, outsOffsets));
  result.addAttribute(
      getOperandSegmentSizesAttrName(result.name),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(boundsOps.size()),
                                    static_cast<int32_t>(insOps.size()),
                                    static_cast<int32_t>(outsOps.size())}));
  result.addTypes(resultTypes);
  return success();
}

void HCGenericOp::print(OpAsmPrinter &p) {
  p << " iter (";
  llvm::interleaveComma(
      llvm::zip_equal(getIterSymsAttr().getAsRange<StringAttr>(),
                      getIterKindsAttr().getAsRange<IterKindAttr>(),
                      getIterBounds()),
      p, [&](auto t) {
        auto [sym, kind, bound] = t;
        p << stringifyIterKind(kind.getValue()) << ' ' << sym.getValue()
          << " = " << bound << " : " << bound.getType();
      });
  p << ")";

  printGenericOperandClause(p, "ins", getIns(), getInsOffsetsAttr());
  printGenericOperandClause(p, "outs", getOuts(), getOutsOffsetsAttr());

  p << " -> (";
  llvm::interleaveComma(getResultTypes(), p, [&](Type t) { p.printType(t); });
  p << ")";

  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getIterSymsAttrName(), getIterKindsAttrName(),
                       getInsOffsetsAttrName(), getOutsOffsetsAttrName(),
                       getOperandSegmentSizesAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true);
}

LogicalResult HCGenericOp::verify() {
  ArrayAttr iterSyms = getIterSymsAttr();
  ArrayAttr iterKinds = getIterKindsAttr();
  OperandRange iterBounds = getIterBounds();
  if (iterSyms.size() != iterBounds.size() ||
      iterSyms.size() != iterKinds.size())
    return emitOpError("iter_syms, iter_bounds, and iter_kinds must agree on "
                       "iter count, got ")
           << iterSyms.size() << ", " << iterBounds.size() << ", "
           << iterKinds.size();
  if (iterSyms.empty())
    return emitOpError("must declare at least one iter");

  llvm::StringSet<> seenSyms;
  llvm::SmallDenseSet<StringRef> reductionSyms;
  for (auto [symAttr, kindAttr] :
       llvm::zip_equal(iterSyms.getAsRange<StringAttr>(),
                       iterKinds.getAsRange<IterKindAttr>())) {
    StringRef name = symAttr.getValue();
    if (name.empty())
      return emitOpError("iter sym names must be non-empty");
    if (!seenSyms.insert(name).second)
      return emitOpError("duplicate iter sym '") << name << "'";
    if (kindAttr.getValue() == IterKind::Reduction)
      reductionSyms.insert(name);
  }

  ArrayAttr insOffsets = getInsOffsetsAttr();
  ArrayAttr outsOffsets = getOutsOffsetsAttr();
  if (insOffsets.size() != getIns().size())
    return emitOpError("ins_offsets count ")
           << insOffsets.size() << " != ins count " << getIns().size();
  if (outsOffsets.size() != getOuts().size())
    return emitOpError("outs_offsets count ")
           << outsOffsets.size() << " != outs count " << getOuts().size();

  if (getOuts().empty())
    return emitOpError("must declare at least one output");

  // SSA results track value-typed outs in declaration order. Ptr/buffer
  // outs land their work in memory through the implicit ptr_store on the
  // yield (op-level effects say so) and contribute no result. A pure-store
  // generic produces zero results; mixed produces one per value-typed out.
  SmallVector<Value> valueOuts;
  for (Value out : getOuts())
    if (!isMemoryCarrierOperand(out.getType()))
      valueOuts.push_back(out);
  if (getResults().size() != valueOuts.size())
    return emitOpError("results count ")
           << getResults().size() << " != value-typed outs count "
           << valueOuts.size() << " (ptr/buffer outs contribute no SSA result)";
  for (auto [i, resTy, outVal] : llvm::enumerate(getResultTypes(), valueOuts)) {
    if (resTy != outVal.getType())
      return emitOpError("result #")
             << i << " type " << resTy << " does not match outs operand type "
             << outVal.getType();
  }

  // Per-operand offset arrays must agree on length with the operand's
  // rank. `!hc.undef` operands have no shape — skip them; the bounds
  // pass / inference fills the rank in once a concrete type lands.
  // Each per-operand offset entry has to be a `#hc.expr` array. We
  // intentionally do NOT enforce `entries.size() == operand.rank`:
  // post-flatten an operand sits on its 1D storage shape but the
  // per-axis offset array still describes the original nD logical
  // access ("ops maintain their own nested structure" across the
  // type-only flatten boundary). A richer post-flatten verifier
  // that pins offsets-vs-iter-syms / offsets-vs-logical-shape lives
  // in a follow-up.
  auto checkPerOperandShape = [&](ArrayAttr perOperandOffsets, StringRef role,
                                  size_t roleIdx) -> LogicalResult {
    auto entries = llvm::dyn_cast<ArrayAttr>(perOperandOffsets);
    if (!entries)
      return emitOpError(role)
             << "_offsets[" << roleIdx
             << "] must be an array of #hc.expr<...> per axis";
    return success();
  };
  for (auto [i, off] : llvm::enumerate(insOffsets.getAsRange<ArrayAttr>()))
    if (failed(checkPerOperandShape(off, "ins", i)))
      return failure();
  for (auto [i, off] : llvm::enumerate(outsOffsets.getAsRange<ArrayAttr>()))
    if (failed(checkPerOperandShape(off, "outs", i)))
      return failure();

  // Reduction iters on output offsets would mean writing the same slot
  // twice along the reduction without specifying a combinator — the op
  // doesn't model that; the reduction value rides on the outs-as-init
  // body-arg/yield channel instead. Walk every per-axis entry.
  for (auto [i, perOperand] :
       llvm::enumerate(outsOffsets.getAsRange<ArrayAttr>())) {
    StringRef bad;
    int64_t badAxis = -1;
    for (auto [axis, axisAttr] :
         llvm::enumerate(perOperand.getAsRange<ExprAttr>())) {
      if (!bad.empty())
        break;
      sym::walkSymbolNames(axisAttr.getValue(), [&](StringRef name) {
        if (!bad.empty())
          return;
        if (reductionSyms.contains(name)) {
          bad = name;
          badAxis = static_cast<int64_t>(axis);
        }
      });
    }
    if (!bad.empty())
      return emitOpError("output #")
             << i << " axis " << badAxis
             << " offset references reduction iter '" << bad
             << "'; only parallel iters may appear in output addressing";
  }

  Region &body = getBody();
  if (body.empty())
    return emitOpError("expected a body region with an entry block");
  Block &entry = body.front();
  size_t expectedArgs = getIns().size() + getOuts().size();
  if (entry.getNumArguments() != expectedArgs)
    return emitOpError("body block takes ")
           << entry.getNumArguments() << " argument(s), expected "
           << expectedArgs << " (one per ins/outs)";

  auto checkArg = [&](size_t blockIdx, Value operand, StringRef role,
                      size_t roleIdx) -> LogicalResult {
    Type elem = genericOperandElement(operand.getType());
    if (!elem)
      return success();
    Type blockArgType = entry.getArgument(blockIdx).getType();
    if (isHCUndefType(blockArgType) || blockArgType == elem)
      return success();
    return emitOpError("body argument #")
           << blockIdx << " type " << blockArgType << " does not match " << role
           << " #" << roleIdx << " element type " << elem;
  };
  for (auto [i, in] : llvm::enumerate(getIns()))
    if (failed(checkArg(i, in, "ins", i)))
      return failure();
  for (auto [i, out] : llvm::enumerate(getOuts()))
    if (failed(checkArg(getIns().size() + i, out, "outs", i)))
      return failure();

  Operation *terminator = tryGetTerminator(entry);
  auto yield = llvm::dyn_cast_or_null<HCYieldOp>(terminator);
  if (!yield)
    return emitOpError("body must terminate with `hc.yield`");
  if (yield.getValues().size() != getOuts().size())
    return emitOpError("hc.yield arity ")
           << yield.getValues().size() << " != outs count " << getOuts().size();
  for (auto [i, yv, outVal] : llvm::enumerate(yield.getValues(), getOuts())) {
    Type yieldType = yv.getType();
    Type outElem = genericOperandElement(outVal.getType());
    if (!outElem || isHCUndefType(yieldType) || yieldType == outElem)
      continue;
    return emitOpError("hc.yield #")
           << i << " type " << yieldType << " does not match outs #" << i
           << " element type " << outElem;
  }

  return success();
}

// Per-operand effects: ptr/buffer in `ins` is a Read of the operand,
// ptr/buffer in `outs` is Read+Write (the read sources the carry —
// outs-as-init contract on a memory destination). Value-typed slots
// contribute nothing, so a generic with all-value outs is pure on
// memory and effect-aware passes (CSE, LICM, speculation) can move it
// freely.
void HCGenericOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  auto *resource = SideEffects::DefaultResource::get();
  for (OpOperand &in : getInsMutable()) {
    if (isMemoryCarrierOperand(in.get().getType()))
      effects.emplace_back(MemoryEffects::Read::get(), &in, resource);
  }
  for (OpOperand &out : getOutsMutable()) {
    if (isMemoryCarrierOperand(out.get().getType())) {
      effects.emplace_back(MemoryEffects::Read::get(), &out, resource);
      effects.emplace_back(MemoryEffects::Write::get(), &out, resource);
    }
  }
}
