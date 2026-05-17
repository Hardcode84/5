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

// Forward decls for `custom<...>` directives. Definitions below the generated
// include to use generated classes.
static mlir::ParseResult parseHCAsLayoutAttr(mlir::OpAsmParser &parser,
                                             mlir::hc::LayoutAttr &layout);
static void printHCAsLayoutAttr(mlir::OpAsmPrinter &printer,
                                mlir::Operation *op,
                                mlir::hc::LayoutAttr layout);
static mlir::ParseResult parseApplyBindings(
    mlir::OpAsmParser &parser,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    mlir::ArrayAttr &symbols);
static void printApplyBindings(mlir::OpAsmPrinter &p, mlir::Operation *op,
                               mlir::OperandRange operands,
                               mlir::ArrayAttr symbols);

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

// Verifier-safe terminator accessor: null on malformed IR (no assert).
static Operation *tryGetTerminator(Block &block) {
  if (block.empty() || !block.mightHaveTerminator())
    return nullptr;
  return block.getTerminator();
}

//===----------------------------------------------------------------------===//
// Shared signature parse/print/verify for hc.kernel/hc.func/hc.intrinsic.
// Surface: `@name (%a: T, ...) (-> T)?`. Block args 1:1 with function_type
// inputs. Signature optional; bare `hc.func @foo { ... }` needs no-arg body.
//===----------------------------------------------------------------------===//

// hc.as_layout's $layout operand. The `(...)` wrapper exists because
// parseExtendedAttr eats a trailing `: type` after a dialect-prefixed attr.
static ParseResult parseHCAsLayoutAttr(OpAsmParser &parser,
                                       LayoutAttr &layout) {
  if (parser.parseLParen() || parser.parseAttribute(layout) ||
      parser.parseRParen())
    return failure();
  return success();
}

static void printHCAsLayoutAttr(OpAsmPrinter &printer, Operation *op,
                                LayoutAttr layout) {
  printer << "(";
  printer.printAttribute(layout);
  printer << ")";
}

// `(%a0: T, ...)` arg list; caller consumed leading `(`. Empty `()` allowed.
static ParseResult
parseSignatureArgList(OpAsmParser &parser,
                      SmallVectorImpl<OpAsmParser::Argument> &arguments) {
  if (succeeded(parser.parseOptionalRParen()))
    return success();
  if (failed(parser.parseArgumentList(arguments, AsmParser::Delimiter::None,
                                      /*allowType=*/true,
                                      /*allowAttrs=*/false)))
    return failure();
  return parser.parseRParen();
}

// Optional `-> T`, `-> (T0, T1)`, or `-> ()` result clause.
static ParseResult
parseOptionalResultTypes(OpAsmParser &parser,
                         SmallVectorImpl<Type> &resultTypes) {
  if (failed(parser.parseOptionalArrow()))
    return success();
  if (failed(parser.parseOptionalLParen())) {
    Type ty;
    if (failed(parser.parseType(ty)))
      return failure();
    resultTypes.push_back(ty);
    return success();
  }
  if (succeeded(parser.parseOptionalRParen()))
    return success();
  if (failed(parser.parseTypeList(resultTypes)))
    return failure();
  return parser.parseRParen();
}

// Optional `(%a: T, ...) (-> T)?` signature. Absent leading `(` leaves both
// outputs default so the no-signature form survives.
static ParseResult parseOptionalFunctionSignature(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::Argument> &arguments,
    TypeAttr &functionTypeAttr) {
  if (!succeeded(parser.parseOptionalLParen()))
    return success();
  if (failed(parseSignatureArgList(parser, arguments)))
    return failure();
  SmallVector<Type> resultTypes;
  if (failed(parseOptionalResultTypes(parser, resultTypes)))
    return failure();
  SmallVector<Type> inputTypes;
  inputTypes.reserve(arguments.size());
  for (auto &arg : arguments)
    inputTypes.push_back(arg.type);
  auto fnType = FunctionType::get(parser.getContext(), inputTypes, resultTypes);
  functionTypeAttr = TypeAttr::get(fnType);
  return success();
}

// Inverse of parseOptionalFunctionSignature. Null functionTypeAttr elides.
// Arg names from entry block for round-trip.
static void printOptionalFunctionSignature(OpAsmPrinter &p, Operation *op,
                                           TypeAttr functionTypeAttr,
                                           Region &body) {
  if (!functionTypeAttr)
    return;
  auto fnType = llvm::cast<FunctionType>(functionTypeAttr.getValue());
  p << '(';
  // Mid-construction IR can violate the entry-block-matches-inputs invariant;
  // print types-only so the printer never derefs a missing block.
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

// Read attr-dict + body; back-fill entry block on `{}` to keep SizedRegion<1>.
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

// Signature-carrying verify: function_type -> entry block args 1:1 with inputs;
// absent -> no args.
static LogicalResult verifyFunctionSignature(Operation *op,
                                             TypeAttr functionTypeAttr,
                                             Region &body) {
  // Hand-built ops can bypass SizedRegion<1>; diagnose instead of asserting.
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

  // Inline `requirements = ...`; elide from attr-dict.
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

// Parse `<keyword> = <attr>` (typed AttrT) onto `attrName`. Absent = no-op.
template <typename AttrT>
static ParseResult
parseOptionalKeywordCustomAttr(OpAsmParser &parser, StringRef keyword,
                               StringAttr attrName, OperationState &result) {
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();
  if (parser.parseEqual())
    return failure();
  AttrT attr;
  if (parser.parseCustomAttributeWithFallback(attr, Type{}))
    return failure();
  result.addAttribute(attrName, attr);
  return success();
}

// Same, for builtin ArrayAttr.
static ParseResult parseOptionalKeywordArrayAttr(OpAsmParser &parser,
                                                 StringRef keyword,
                                                 StringAttr attrName,
                                                 OperationState &result) {
  if (failed(parser.parseOptionalKeyword(keyword)))
    return success();
  if (parser.parseEqual())
    return failure();
  ArrayAttr attr;
  if (parser.parseAttribute(attr))
    return failure();
  result.addAttribute(attrName, attr);
  return success();
}

// Required `scope = #hc.scope<...>` clause.
static ParseResult parseIntrinsicScopeClause(OpAsmParser &parser,
                                             OperationState &result) {
  if (parser.parseKeyword("scope") || parser.parseEqual())
    return failure();
  ScopeAttr scope;
  if (parser.parseCustomAttributeWithFallback(scope, Type{}))
    return failure();
  result.addAttribute(HCIntrinsicOp::getScopeAttrName(result.name), scope);
  return success();
}

// Optional effects/const_kwargs/parameters/keyword_only in declaration order.
static ParseResult parseIntrinsicMetadataKeywords(OpAsmParser &parser,
                                                  OperationState &result) {
  if (parseOptionalKeywordCustomAttr<EffectClassAttr>(
          parser, "effects", HCIntrinsicOp::getEffectsAttrName(result.name),
          result))
    return failure();
  if (parseOptionalKeywordArrayAttr(
          parser, "const_kwargs",
          HCIntrinsicOp::getConstKwargsAttrName(result.name), result))
    return failure();
  if (parseOptionalKeywordArrayAttr(
          parser, "parameters",
          HCIntrinsicOp::getParametersAttrName(result.name), result))
    return failure();
  return parseOptionalKeywordArrayAttr(
      parser, "keyword_only",
      HCIntrinsicOp::getKeywordOnlyAttrName(result.name), result);
}

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

  if (parseIntrinsicScopeClause(parser, result) ||
      parseIntrinsicMetadataKeywords(parser, result))
    return failure();

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

// No parameters: const_kwargs/keyword_only forbidden, function_type has 0
// inputs.
static LogicalResult verifyIntrinsicWithoutParameters(HCIntrinsicOp op,
                                                      TypeAttr fnTypeAttr) {
  if (op.getConstKwargsAttr())
    return op.emitOpError("const_kwargs requires parameters to declare the "
                          "full intrinsic parameter order");
  if (op.getKeywordOnlyAttr())
    return op.emitOpError("keyword_only requires parameters to declare the "
                          "full intrinsic parameter order");
  if (!fnTypeAttr)
    return success();
  auto fnType = llvm::cast<FunctionType>(fnTypeAttr.getValue());
  if (fnType.getNumInputs() != 0)
    return op.emitOpError("function_type with inputs requires parameters to "
                          "name the intrinsic operand order");
  return success();
}

// parameters: each entry a unique non-empty StringAttr.
static LogicalResult
verifyIntrinsicParameterEntries(HCIntrinsicOp op, ArrayAttr parameters,
                                llvm::SmallDenseSet<StringRef> &declared) {
  for (auto [idx, parameter] : llvm::enumerate(parameters)) {
    auto name = dyn_cast<StringAttr>(parameter);
    if (!name)
      return op.emitOpError("parameters entry at index ")
             << idx << " must be a StringAttr, got " << parameter;
    if (!declared.insert(name.getValue()).second)
      return op.emitOpError("duplicate parameter name '")
             << name.getValue() << "'";
  }
  return success();
}

// keyword_only: names must appear in declared.
static LogicalResult
verifyIntrinsicKeywordOnly(HCIntrinsicOp op,
                           const llvm::SmallDenseSet<StringRef> &declared,
                           llvm::SmallDenseSet<StringRef> &keywordOnlyNames) {
  ArrayAttr keywordOnly = op.getKeywordOnlyAttr();
  if (!keywordOnly)
    return success();
  for (Attribute kw : keywordOnly) {
    auto kwName = dyn_cast<StringAttr>(kw);
    if (!kwName)
      return op.emitOpError("keyword_only entry must be a StringAttr, got ")
             << kw;
    StringRef name = kwName.getValue();
    if (!keywordOnlyNames.insert(name).second)
      return op.emitOpError("duplicate keyword_only entry '") << name << "'";
    if (!declared.contains(name))
      return op.emitOpError("keyword_only entry '")
             << name << "' is not listed in parameters";
  }
  return success();
}

// Positional names must precede first keyword-only.
static LogicalResult verifyIntrinsicPositionalOrder(
    HCIntrinsicOp op, ArrayAttr parameters,
    const llvm::SmallDenseSet<StringRef> &keywordOnlyNames) {
  bool seenKeywordOnly = false;
  for (Attribute parameter : parameters) {
    StringRef name = cast<StringAttr>(parameter).getValue();
    if (keywordOnlyNames.contains(name)) {
      seenKeywordOnly = true;
      continue;
    }
    if (seenKeywordOnly)
      return op.emitOpError("positional parameter '")
             << name << "' cannot follow a keyword-only parameter";
  }
  return success();
}

// const_kwargs: each in declared and keyword_only (not SSA operands).
static LogicalResult verifyIntrinsicConstKwargs(
    HCIntrinsicOp op, const llvm::SmallDenseSet<StringRef> &declared,
    const llvm::SmallDenseSet<StringRef> &keywordOnlyNames) {
  ArrayAttr constKwargs = op.getConstKwargsAttr();
  if (!constKwargs)
    return success();
  llvm::SmallDenseSet<StringRef> seenConstKwargs;
  for (Attribute kw : constKwargs) {
    auto kwName = dyn_cast<StringAttr>(kw);
    if (!kwName)
      return op.emitOpError("const_kwargs entry must be a StringAttr, got ")
             << kw;
    StringRef name = kwName.getValue();
    if (!seenConstKwargs.insert(name).second)
      return op.emitOpError("duplicate const_kwargs entry '") << name << "'";
    if (!declared.contains(name))
      return op.emitOpError("const_kwargs entry '")
             << name << "' is not listed in parameters";
    if (!keywordOnlyNames.contains(name))
      return op.emitOpError("const_kwargs entry '")
             << name << "' must be listed in keyword_only";
  }
  return success();
}

LogicalResult HCIntrinsicOp::verify() {
  if (failed(verifyFunctionSignature(*this, getFunctionTypeAttr(), getBody())))
    return failure();

  ArrayAttr parameters = getParametersAttr();
  TypeAttr fnTypeAttr = getFunctionTypeAttr();
  if (!parameters)
    return verifyIntrinsicWithoutParameters(*this, fnTypeAttr);

  llvm::SmallDenseSet<StringRef> declared;
  if (failed(verifyIntrinsicParameterEntries(*this, parameters, declared)))
    return failure();

  if (!fnTypeAttr)
    return emitOpError(
        "parameters requires function_type to define the runtime SSA "
        "operand signature");

  llvm::SmallDenseSet<StringRef> keywordOnlyNames;
  if (failed(verifyIntrinsicKeywordOnly(*this, declared, keywordOnlyNames)) ||
      failed(verifyIntrinsicPositionalOrder(*this, parameters,
                                            keywordOnlyNames)) ||
      failed(verifyIntrinsicConstKwargs(*this, declared, keywordOnlyNames)))
    return failure();

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
// hc.return. Optional terminator matching enclosing callable. Kernels: empty.
// Funcs/intrinsics: types match declared results.
//===----------------------------------------------------------------------===//

// Nearest enclosing callable; subgroup_region/workitem_region/for_range/if
// are transparent.
static Operation *findReturnEnclosingCallee(HCReturnOp op) {
  Operation *callee = op->getParentOp();
  while (callee && !isa<HCKernelOp, HCFuncOp, HCIntrinsicOp>(callee))
    callee = callee->getParentOp();
  return callee;
}

// Returned types match declared results.
static LogicalResult verifyReturnAgainstFunctionType(HCReturnOp op,
                                                     Operation *callee,
                                                     FunctionType fnType) {
  if (op.getValues().size() != fnType.getNumResults())
    return op.emitOpError("returns ")
           << op.getValues().size() << " value(s) but enclosing "
           << callee->getName() << " declares " << fnType.getNumResults()
           << " result(s)";
  for (auto [i, returned, declared] :
       llvm::enumerate(op.getValues().getTypes(), fnType.getResults())) {
    if (!areHCProgressiveTypesCompatible(returned, declared))
      return op.emitOpError("returned value #")
             << i << " type " << returned << " does not match enclosing "
             << callee->getName() << " result type " << declared;
  }
  return success();
}

LogicalResult HCReturnOp::verify() {
  Operation *callee = findReturnEnclosingCallee(*this);
  if (!callee)
    return success();

  // Kernels return no values.
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
  return verifyReturnAgainstFunctionType(*this, callee, fnType);
}

LogicalResult HCSymbolOp::verify() {
  // ODS pins !hc.idx; pin the expression. Unpinned !hc.idx is an inferred
  // unbound capture, not a binding.
  if (!llvm::cast<IdxType>(getResult().getType()).getExpr())
    return emitOpError("result must pin a symbolic expression "
                       "(e.g. `!hc.idx<\"M\">`)");
  return success();
}

// Inline bindings for hc.idx_apply / hc.pred_apply:
//   hc.idx_apply (%a as "i", %b as "j") : (index, index) -> !hc.idx<"i + j">
//   hc.idx_apply () : () -> !hc.idx<"M">     // free syms ambient
// Captures (op, name) pairs; operand types from trailing functional-type.
static ParseResult
parseApplyBindings(OpAsmParser &parser,
                   SmallVectorImpl<OpAsmParser::UnresolvedOperand> &operands,
                   ArrayAttr &symbols) {
  SmallVector<Attribute> symAttrs;
  auto parseOne = [&]() -> ParseResult {
    OpAsmParser::UnresolvedOperand operand;
    StringAttr name;
    if (parser.parseOperand(operand) || parser.parseKeyword("as") ||
        parser.parseAttribute(name))
      return failure();
    operands.push_back(operand);
    symAttrs.push_back(name);
    return success();
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseOne))
    return failure();
  symbols = ArrayAttr::get(parser.getContext(), symAttrs);
  return success();
}

static void printApplyBindings(OpAsmPrinter &p, Operation * /*op*/,
                               OperandRange operands, ArrayAttr symbols) {
  p << '(';
  llvm::interleaveComma(llvm::zip(operands, symbols), p, [&](auto pair) {
    p << std::get<0>(pair) << " as " << std::get<1>(pair);
  });
  p << ')';
}

// Shared check: symbols and operands line up, names unique, every listed name
// is free in payload. Unlisted free syms stay ambient (launch context).
template <typename WalkFreeSyms>
static LogicalResult verifySymBindings(Operation *op, size_t numOperands,
                                       ArrayAttr symbolsAttr,
                                       WalkFreeSyms walkFreeSyms) {
  if (symbolsAttr.size() != numOperands)
    return op->emitOpError("symbols list has ")
           << symbolsAttr.size() << " entries but the op has " << numOperands
           << " operand(s)";

  llvm::StringSet<> freeSyms;
  walkFreeSyms([&](StringRef name) { freeSyms.insert(name); });

  llvm::StringSet<> seen;
  for (auto [idx, attr] : llvm::enumerate(symbolsAttr)) {
    auto str = llvm::dyn_cast<StringAttr>(attr);
    if (!str)
      return op->emitOpError("symbols entry #")
             << idx << " is not a StringAttr";
    StringRef name = str.getValue();
    if (name.empty())
      return op->emitOpError("symbols entry #") << idx << " is empty";
    if (!seen.insert(name).second)
      return op->emitOpError("duplicate symbol binding for '") << name << "'";
    if (!freeSyms.contains(name))
      return op->emitOpError("symbol '")
             << name
             << "' is not a free symbol of the carried expression / predicate";
  }
  return success();
}

LogicalResult HCIdxApplyOp::verify() {
  auto idx = llvm::dyn_cast<IdxType>(getResult().getType());
  if (!idx || !idx.getExpr())
    return emitOpError("result must pin a symbolic expression "
                       "(e.g. `!hc.idx<\"i*K + j\">`)");
  ExprAttr expr = idx.getExpr();
  return verifySymBindings(*this, getOperands().size(), getSymbolsAttr(),
                           [&](llvm::function_ref<void(StringRef)> sink) {
                             sym::walkSymbolNames(expr.getValue(), sink);
                           });
}

LogicalResult HCPredApplyOp::verify() {
  auto pred = llvm::dyn_cast<PredType>(getResult().getType());
  if (!pred || !pred.getPred())
    return emitOpError("result must pin a symbolic predicate "
                       "(e.g. `!hc.pred<\"i < K\">`)");
  PredAttr predicate = pred.getPred();
  return verifySymBindings(*this, getOperands().size(), getSymbolsAttr(),
                           [&](llvm::function_ref<void(StringRef)> sink) {
                             sym::walkSymbolNames(predicate.getValue(), sink);
                           });
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
  // Symbolic bounds; trip count unknown. Model zero-trip and body.
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

// Per-axis suffix dims -> single #hc.expr product via ixsimpl. Empty = 1.
// Each entry must be ExprAttr; collectiveSuffix enforces upstream.
static FailureOr<ExprAttr> composeSuffixProduct(MLIRContext *ctx,
                                                ArrayRef<Attribute> suffix) {
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  if (suffix.empty()) {
    auto one = sym::composeExprInt(store, 1);
    if (failed(one))
      return failure();
    return ExprAttr::get(ctx, *one);
  }
  auto first = dyn_cast<ExprAttr>(suffix.front());
  if (!first)
    return failure();
  sym::ExprHandle product = first.getValue();
  for (Attribute dim : suffix.drop_front()) {
    auto next = dyn_cast<ExprAttr>(dim);
    if (!next)
      return failure();
    auto handle = sym::composeExprBinary(store, product, sym::ExprBinaryOp::Mul,
                                         next.getValue());
    if (failed(handle))
      return failure();
    product = *handle;
  }
  return ExprAttr::get(ctx, product);
}

// Storage #hc.expr from 1D shaped type (post-flatten: no layout, single dim).
// Failure -> not post-flatten.
static FailureOr<ExprAttr> postFlattenStorageExpr(Type type) {
  auto shaped = dyn_cast<SymbolicallyShapedTypeInterface>(type);
  if (!shaped)
    return failure();
  if (shaped.getSymbolicLayout())
    return failure();
  ShapeAttr shape = shaped.getSymbolicShape();
  if (!shape)
    return failure();
  ArrayRef<Attribute> dims = shape.getDims();
  if (dims.size() != 1)
    return failure();
  auto expr = dyn_cast<ExprAttr>(dims.front());
  if (!expr)
    return failure();
  return expr;
}

// Post-flatten lift: result_storage == yield_storage * product(suffix).
// Both sides 1D bare, same kind, same element. Tuples recurse pointwise;
// scalar yield carries yield_storage=1. Nested tuples not modeled.
static bool postFlattenLiftMatchesTuple(TupleType yt, Type resultType,
                                        ArrayRef<Attribute> suffix) {
  auto rt = dyn_cast<TupleType>(resultType);
  if (!rt || yt.size() != rt.size())
    return false;
  for (auto [yElem, rElem] : llvm::zip_equal(yt.getTypes(), rt.getTypes())) {
    if (isa<TupleType>(yElem))
      return false;
    if (!mlir::hc::postFlattenLiftMatches(yElem, rElem, suffix))
      return false;
  }
  return true;
}

// Yield-side storage for post-flatten lift. Failure on incompatible kinds.
static FailureOr<ExprAttr>
postFlattenLiftYieldStorage(Type yieldedType, Type resultType,
                            SymbolicallyShapedTypeInterface resultShaped,
                            sym::Store &store) {
  MLIRContext *ctx = yieldedType.getContext();
  if (auto yieldShaped =
          dyn_cast<SymbolicallyShapedTypeInterface>(yieldedType)) {
    if (yieldedType.getTypeID() != resultType.getTypeID())
      return failure();
    if (resultShaped.getSymbolicElementType() !=
        yieldShaped.getSymbolicElementType())
      return failure();
    return postFlattenStorageExpr(yieldedType);
  }
  if (!isCollectiveScalarType(yieldedType))
    return failure();
  // Scalar lift wraps in hc::VectorType pre-flatten; bare-vector illegal here.
  if (!isa<mlir::hc::VectorType>(resultType) ||
      resultShaped.getSymbolicElementType() != yieldedType)
    return failure();
  auto one = sym::composeExprInt(store, 1);
  if (failed(one))
    return failure();
  return ExprAttr::get(ctx, *one);
}

bool mlir::hc::postFlattenLiftMatches(Type yieldedType, Type resultType,
                                      ArrayRef<Attribute> suffix) {
  if (auto yt = dyn_cast<TupleType>(yieldedType))
    return postFlattenLiftMatchesTuple(yt, resultType, suffix);

  FailureOr<ExprAttr> resultStorage = postFlattenStorageExpr(resultType);
  if (failed(resultStorage))
    return false;
  auto resultShaped = cast<SymbolicallyShapedTypeInterface>(resultType);

  MLIRContext *ctx = yieldedType.getContext();
  auto &store = ctx->getOrLoadDialect<HCDialect>()->getSymbolStore();
  FailureOr<ExprAttr> yieldStorage =
      postFlattenLiftYieldStorage(yieldedType, resultType, resultShaped, store);
  if (failed(yieldStorage))
    return false;

  FailureOr<ExprAttr> suffixProduct = composeSuffixProduct(ctx, suffix);
  if (failed(suffixProduct))
    return false;
  auto lifted =
      sym::composeExprBinary(store, yieldStorage->getValue(),
                             sym::ExprBinaryOp::Mul, suffixProduct->getValue());
  if (failed(lifted))
    return false;
  return ExprAttr::get(ctx, *lifted) == *resultStorage;
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
  // Pre-flatten: structural append of suffix; literal type equality.
  if (collectiveLiftedType(yieldedType, *suffix) == resultType)
    return true;
  // Post-flatten: 1D bare; result.storage == yield.storage * product(suffix).
  return postFlattenLiftMatches(yieldedType, resultType, *suffix);
}

// iter_inits/result counts + pairwise types; body shape: induction_var,
// iter_args.
static LogicalResult verifyForRangeIterSignature(HCForRangeOp op) {
  Block &body = op.getBody().front();
  unsigned expectedArgs = 1 + op.getIterInits().size();
  if (body.getNumArguments() != expectedArgs)
    return op.emitOpError("expected body block to take ")
           << expectedArgs << " arguments (induction variable + "
           << op.getIterInits().size() << " iter_args), got "
           << body.getNumArguments();
  if (op.getIterResults().size() != op.getIterInits().size())
    return op.emitOpError("iter_args (")
           << op.getIterInits().size() << ") and result count ("
           << op.getIterResults().size() << ") differ";
  for (auto [idx, pair] : llvm::enumerate(
           llvm::zip_equal(op.getIterInits(), op.getIterResults()))) {
    auto [init, result] = pair;
    if (!areHCBranchTypesCompatible(init.getType(), result.getType()))
      return op.emitOpError("iter_args[")
             << idx << "] type " << init.getType() << " does not match result["
             << idx << "] type " << result.getType();
  }
  return success();
}

// Block args 1:1 with iter_inits past induction var. `!hc.undef` matches
// freely.
static LogicalResult verifyForRangeBlockArgTypes(HCForRangeOp op) {
  Block &body = op.getBody().front();
  for (auto [idx, pair] : llvm::enumerate(llvm::zip_equal(
           op.getIterInits(), body.getArguments().drop_front(1)))) {
    auto [init, blockArg] = pair;
    Type initTy = init.getType();
    Type blockTy = blockArg.getType();
    if (areHCBranchTypesCompatible(initTy, blockTy))
      continue;
    return op.emitOpError("iter_args[")
           << idx << "] type " << initTy
           << " does not match body block argument type " << blockTy;
  }
  return success();
}

// Body must terminate hc.yield; operands match iter_results.
static LogicalResult verifyForRangeYield(HCForRangeOp op) {
  Block &body = op.getBody().front();
  auto yield = llvm::dyn_cast_or_null<HCYieldOp>(tryGetTerminator(body));
  if (!yield)
    return op.emitOpError("body must terminate with an `hc.yield`");
  if (yield.getValues().size() != op.getIterResults().size())
    return op.emitOpError("body yield produces ")
           << yield.getValues().size() << " values, expected "
           << op.getIterResults().size();
  for (auto [idx, pair] : llvm::enumerate(
           llvm::zip_equal(yield.getValues(), op.getIterResults()))) {
    auto [yielded, result] = pair;
    Type yieldedTy = yielded.getType();
    Type resultTy = result.getType();
    if (areHCBranchTypesCompatible(yieldedTy, resultTy))
      continue;
    return op.emitOpError("body yield[")
           << idx << "] type " << yieldedTy << " does not match result[" << idx
           << "] type " << resultTy;
  }
  return success();
}

LogicalResult HCForRangeOp::verify() {
  if (failed(verifyForRangeIterSignature(*this)) ||
      failed(verifyForRangeBlockArgTypes(*this)) ||
      failed(verifyForRangeYield(*this)))
    return failure();
  return success();
}

// Shared verifier for hc.workitem_region/hc.subgroup_region.
// Empty $results: pre-promotion, terminator unconstrained.
// Non-empty: body ends hc.yield, arity + types compatible. Collective regions
// also accept the lift: yielded scalar/vector gains participant suffix.
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

// $names entries name the writebacks post promote-names. Dups alias.
LogicalResult HCRegionReturnOp::verify() {
  llvm::SmallPtrSet<StringAttr, 4> seen;
  for (auto [idx, raw] : llvm::enumerate(getNames())) {
    auto name = llvm::dyn_cast<StringAttr>(raw);
    if (!name)
      return emitOpError("`names[")
             << idx << "]` is not a StringAttr (got " << raw << ")";
    if (!seen.insert(name).second)
      return emitOpError("duplicate name '")
             << name.getValue() << "' in `names`; duplicates alias";
  }
  return success();
}

LogicalResult HCIfOp::verify() {
  // Non-empty region yield matches result types. Empty else only when no
  // results.
  auto checkRegion = [&](Region &region,
                         llvm::StringRef label) -> LogicalResult {
    if (region.empty())
      return success();
    // tryGetTerminator returns null on malformed regions: error > crash.
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

/// Concrete shape attr from symbolically shaped type; null pre-inference.
static ShapeAttr tryGetShape(Type t) {
  if (auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(t))
    return shaped.getSymbolicShape();
  return nullptr;
}

/// First enclosing subgroup/workitem region, or null in workgroup scope.
/// Kernel/func/intrinsic/symbol-table rebaseline scope.
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

/// Tensor allocators are workgroup-scope only.
static LogicalResult verifyTensorAllocScope(Operation *op) {
  if (Operation *narrowing = findNarrowingScope(op))
    return op->emitOpError("tensor allocator is workgroup scope only; "
                           "enclosed by ")
           << narrowing->getName() << " which narrows the scope";
  return success();
}

LogicalResult HCBufferDimOp::verify() {
  // Dialect form canonicalizes negative axes away.
  if (getAxisAttr().getValue().isNegative())
    return emitOpError("axis must be non-negative, got ")
           << getAxisAttr().getValue().getSExtValue();
  // Range check needs concrete rank; verifier re-fires post-inference.
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

LogicalResult HCBufferViewOp::verify() {
  // unit_axes are OUTPUT positions for NumPy None unit-dim inserts.
  // Output rank = residual indices + unit-axis count; positions unique, in
  // range.
  auto unitAxes = getUnitAxesAttr();
  if (!unitAxes)
    return success();
  ArrayRef<int64_t> positions = unitAxes.asArrayRef();
  if (positions.empty())
    return success();
  size_t outputRank = getIndices().size() + positions.size();
  llvm::SmallDenseSet<int64_t> seen;
  for (int64_t pos : positions) {
    if (pos < 0)
      return emitOpError("unit_axes entries must be non-negative, got ") << pos;
    if (static_cast<size_t>(pos) >= outputRank)
      return emitOpError("unit_axes entry ")
             << pos << " is out of range for output rank " << outputRank
             << " (= " << getIndices().size() << " residual indices + "
             << positions.size() << " unit axes)";
    if (!seen.insert(pos).second)
      return emitOpError("unit_axes entries must be unique; ")
             << pos << " repeats";
  }
  return success();
}

LogicalResult HCReduceOp::verify() {
  if (getAxisAttr().getValue().isNegative())
    return emitOpError("axis must be non-negative, got ")
           << getAxisAttr().getValue().getSExtValue();
  // Rank-concrete only; pre-inference defers.
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
  // Numeric conversion only; non-builtin numeric target rejected.
  Type target = getTarget();
  if (!target.isIntOrIndexOrFloat())
    return emitOpError("target type must be a builtin integer, index, or "
                       "float type, got ")
           << target;
  // result agrees with target: scalar direct, shaped on element. `!hc.undef`
  // escape for pre-inference.
  Type result = getResult().getType();
  if (isHCUndefType(result))
    return success();
  if (result.isIntOrIndexOrFloat()) {
    if (result != target)
      return emitOpError("result type ")
             << result << " does not match target type " << target;
    return success();
  }
  // Accept semantic and post-decompose bare carriers; decompose splits
  // semantic astype into bare-data astype + mask pass-through.
  auto elementOf = [](Type t) -> Type {
    if (!llvm::isa<mlir::hc::TensorType, mlir::hc::VectorType,
                   mlir::hc::BareTensorType, mlir::hc::BareVectorType>(t))
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
         << " must be `!hc.undef`, a builtin numeric scalar, or a shaped "
            "`!hc.tensor` / `!hc.vector` / `!hc.bare_tensor` / "
            "`!hc.bare_vector` whose element type matches target";
}

LogicalResult HCWithInactiveOp::verify() {
  // Once both operands have inferred domains, $inactive's type must agree with
  // masked value's element type.
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

// Mask carrier shape AND layout must match data; verifiers reconstruct.
static Type barePredicateMaskType(Type type) {
  auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(type);
  if (!shaped)
    return {};
  Type pred = getUnpinnedPredType(type.getContext());
  LayoutAttr layout = shaped.getSymbolicLayout();
  if (llvm::isa<BareTensorType>(type))
    return BareTensorType::get(type.getContext(), pred,
                               shaped.getSymbolicShape(), layout);
  if (llvm::isa<BareVectorType>(type))
    return BareVectorType::get(type.getContext(), pred,
                               shaped.getSymbolicShape(), layout);
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

// Per-side flavor flags for one-sweep strip-layout check.
namespace {
struct StripLayoutFlavorFlags {
  bool isVector;
  bool isBareVector;
  bool isTensor;
  bool isBareTensor;

  bool any() const {
    return isVector || isBareVector || isTensor || isBareTensor;
  }
};
} // namespace

static StripLayoutFlavorFlags makeStripFlavorFlags(Type type) {
  return {isa<mlir::hc::VectorType>(type), isa<mlir::hc::BareVectorType>(type),
          isa<mlir::hc::TensorType>(type), isa<mlir::hc::BareTensorType>(type)};
}

// Strip drops layout, keeps carrier kind. Cross-kind clashes downstream.
static LogicalResult verifyStripFlavorPreservation(HCStripLayoutOp op,
                                                   Type valueType,
                                                   Type resultType) {
  auto src = makeStripFlavorFlags(valueType);
  auto dst = makeStripFlavorFlags(resultType);
  if (src.isVector && !dst.isVector)
    return op.emitOpError("vector operand requires vector result; got ")
           << resultType;
  if (src.isBareVector && !dst.isBareVector)
    return op.emitOpError("bare_vector operand requires bare_vector "
                          "result; got ")
           << resultType;
  if (src.isTensor && !dst.isTensor)
    return op.emitOpError("tensor operand requires tensor result; got ")
           << resultType;
  if (src.isBareTensor && !dst.isBareTensor)
    return op.emitOpError("bare_tensor operand requires bare_tensor "
                          "result; got ")
           << resultType;
  if (!src.any())
    return op.emitOpError("operand must be a vector / tensor (semantic or "
                          "bare); got ")
           << valueType;
  return success();
}

// Strip preserves element + shape; only layout drops.
static LogicalResult
verifyStripElementAndShape(HCStripLayoutOp op,
                           SymbolicallyShapedTypeInterface valueShaped,
                           SymbolicallyShapedTypeInterface resultShaped) {
  Type valueElem = valueShaped.getSymbolicElementType();
  Type resultElem = resultShaped.getSymbolicElementType();
  if (valueElem && resultElem && valueElem != resultElem)
    return op.emitOpError("operand element type ")
           << valueElem << " differs from result element type " << resultElem;

  ShapeAttr valueShape = valueShaped.getSymbolicShape();
  ShapeAttr resultShape = resultShaped.getSymbolicShape();
  if (valueShape && resultShape && valueShape != resultShape)
    return op.emitOpError("operand shape ")
           << valueShape << " differs from result shape " << resultShape
           << "; hc.strip_layout drops the layout, it doesn't reshape";
  return success();
}

// Result layout slot null; neither side may be !hc.buffer (host-side storage,
// reinterpreted via hc.as_layout).
static LogicalResult
verifyStripCarrierShape(HCStripLayoutOp op, Type valueType, Type resultType,
                        SymbolicallyShapedTypeInterface resultShaped) {
  if (resultShaped.getSymbolicLayout())
    return op.emitOpError("result type must have no layout slot; "
                          "use hc.as_layout to relabel a layout-bearing value");
  if (isa<BufferType>(valueType) || isa<BufferType>(resultType))
    return op.emitOpError("hc.strip_layout does not apply to !hc.buffer; "
                          "buffer storage lives host-side, the bare/non-bare "
                          "distinction is value-semantic only");
  return success();
}

LogicalResult HCStripLayoutOp::verify() {
  // Progressive typing: !hc.undef on either side escapes.
  Type valueType = getValue().getType();
  Type resultType = getResult().getType();
  if (!valueType || !resultType || isHCUndefType(valueType) ||
      isHCUndefType(resultType))
    return success();

  auto valueShaped = dyn_cast<SymbolicallyShapedTypeInterface>(valueType);
  auto resultShaped = dyn_cast<SymbolicallyShapedTypeInterface>(resultType);
  if (!valueShaped || !resultShaped)
    return success();

  if (failed(verifyStripCarrierShape(*this, valueType, resultType,
                                     resultShaped)) ||
      failed(verifyStripFlavorPreservation(*this, valueType, resultType)))
    return failure();
  return verifyStripElementAndShape(*this, valueShaped, resultShaped);
}

// as_layout reinterprets layout, not data type.
static LogicalResult
verifyAsLayoutElementType(HCAsLayoutOp op,
                          SymbolicallyShapedTypeInterface valueShaped,
                          SymbolicallyShapedTypeInterface resultShaped) {
  Type valueElem = valueShaped.getSymbolicElementType();
  Type resultElem = resultShaped.getSymbolicElementType();
  if (valueElem && resultElem && valueElem != resultElem)
    return op.emitOpError("operand element type ")
           << valueElem << " differs from result element type " << resultElem
           << "; hc.as_layout reinterprets the layout, not the data type";
  return success();
}

// Storage-size check skipped for buffer (logical extent) and non-ExprAttr.
static LogicalResult
verifyAsLayoutStorageSize(HCAsLayoutOp op, Type valueType, Type resultType,
                          SymbolicallyShapedTypeInterface valueShaped,
                          ShapeAttr valueShape, ShapeAttr resultShape) {
  if (isa<BufferType>(valueType) || isa<BufferType>(resultType))
    return success();

  auto allDimsAreExprs = [](ShapeAttr s) {
    return llvm::all_of(s.getDims(), [](Attribute a) {
      return llvm::isa_and_nonnull<ExprAttr>(a);
    });
  };
  if (!allDimsAreExprs(valueShape) || !allDimsAreExprs(resultShape))
    return success();

  MLIRContext *ctx = op.getContext();
  FailureOr<ExprAttr> valueStorage =
      computeStorageSizeExpr(ctx, valueShaped.getSymbolicLayout(), valueShape);
  if (failed(valueStorage))
    return success();
  FailureOr<ExprAttr> resultStorage =
      computeStorageSizeExpr(ctx, op.getLayout(), resultShape);
  if (failed(resultStorage))
    return success();

  // ixsimpl pointer eq = structural eq. `M*N*4` and `4*N*M` canonicalize same.
  if (*valueStorage != *resultStorage)
    return op.emitOpError("storage size mismatch: operand addresses ")
           << *valueStorage << " elements, result layout addresses "
           << *resultStorage << "; hc.as_layout reinterprets without resizing";
  return success();
}

// `shape=` only for pointer-rooted operands; value-semantic dims would clash.
static LogicalResult verifyAsLayoutShapeOperand(HCAsLayoutOp op,
                                                Type valueType) {
  Value shape = op.getShape();
  if (!shape)
    return success();
  if (valueType && !isHCUndefType(valueType) && !isa<BufferType>(valueType))
    return op.emitOpError(
               "`shape=` operand is only valid on a `!hc.buffer` operand; got ")
           << valueType;
  Type shapeT = shape.getType();
  if (shapeT && !isHCUndefType(shapeT) && !isa<TupleType>(shapeT))
    return op.emitOpError("`shape=` operand must be a tuple, got ") << shapeT;
  return success();
}

// Both shaped: check element, then storage_size when dims pinned.
static LogicalResult
verifyAsLayoutShapedSides(HCAsLayoutOp op, Type valueType, Type resultType,
                          SymbolicallyShapedTypeInterface valueShaped,
                          SymbolicallyShapedTypeInterface resultShaped) {
  if (failed(verifyAsLayoutElementType(op, valueShaped, resultShaped)))
    return failure();
  ShapeAttr valueShape = valueShaped.getSymbolicShape();
  ShapeAttr resultShape = resultShaped.getSymbolicShape();
  if (!valueShape || !resultShape)
    return success();
  return verifyAsLayoutStorageSize(op, valueType, resultType, valueShaped,
                                   valueShape, resultShape);
}

LogicalResult HCAsLayoutOp::verify() {
  // Progressive typing: !hc.undef or unpinned dims escape.
  Type valueType = getValue().getType();
  Type resultType = getResult().getType();
  if (failed(verifyAsLayoutShapeOperand(*this, valueType)))
    return failure();

  if (!valueType || !resultType || isHCUndefType(valueType) ||
      isHCUndefType(resultType))
    return success();

  auto valueShaped = dyn_cast<SymbolicallyShapedTypeInterface>(valueType);
  auto resultShaped = dyn_cast<SymbolicallyShapedTypeInterface>(resultType);
  if (!valueShaped || !resultShaped)
    return success();

  return verifyAsLayoutShapedSides(*this, valueType, resultType, valueShaped,
                                   resultShaped);
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
// hc.ptr_offset/ptr_load/ptr_store: addrspace and (if pinned) elementType
// agree with operand/result. ptr_load/ptr_store accept scalar or vector<NxT>;
// parity on element type. !hc.undef escapes.
//===----------------------------------------------------------------------===//

namespace {

// PtrType payload; null on !hc.undef. HC_PtrValueType rejects others.
static PtrType ptrTypeOrUndef(Type type) {
  if (isHCUndefType(type))
    return {};
  return llvm::cast<PtrType>(type);
}

// Element-type parity for ptr_load/store. Typed ptr: exact match against
// value's (or `vector<NxT>` element) type. Opaque ptr: anything. `!hc.undef`
// escapes; caller has filtered the `!hc.ptr` shell.
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

// Predicated ptr access shape parity. scalar<->i1, vector<NxT><->vector<Nxi1>.
static LogicalResult checkPredicateShapeMatchesValue(Operation *op,
                                                     Type valueType,
                                                     Type predicateType) {
  if (isHCUndefType(valueType) || isHCUndefType(predicateType))
    return success();
  auto valueVec = llvm::dyn_cast<mlir::VectorType>(valueType);
  auto predVec = llvm::dyn_cast<mlir::VectorType>(predicateType);
  if (static_cast<bool>(valueVec) != static_cast<bool>(predVec))
    return op->emitOpError(
               "predicate shape must match value shape: scalar value "
               "requires scalar (i1 / !hc.pred) predicate, vector value "
               "requires vector<...xi1> predicate; got value type ")
           << valueType << " and predicate type " << predicateType;
  if (valueVec && predVec && valueVec.getShape() != predVec.getShape())
    return op->emitOpError("predicate shape ")
           << predicateType << " must match value shape " << valueType;
  return success();
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
  // No piecemeal typed<->opaque drop; typed->opaque only at LLVM-lowering
  // boundary.
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

LogicalResult HCPtrLoadPredOp::verify() {
  PtrType source = ptrTypeOrUndef(getSource().getType());
  if (failed(checkLoadStoreValueMatchesPointer(
          getOperation(), getResult().getType(), source, "result")))
    return failure();
  return checkPredicateShapeMatchesValue(getOperation(), getResult().getType(),
                                         getPredicate().getType());
}

LogicalResult HCPtrStorePredOp::verify() {
  PtrType dest = ptrTypeOrUndef(getDest().getType());
  if (failed(checkLoadStoreValueMatchesPointer(
          getOperation(), getValue().getType(), dest, "value")))
    return failure();
  return checkPredicateShapeMatchesValue(getOperation(), getValue().getType(),
                                         getPredicate().getType());
}

// AllTypesMatch ties $value/$passthrough/$result; only value<->mask shape here.
LogicalResult HCPredicateOp::verify() {
  return checkPredicateShapeMatchesValue(getOperation(), getValue().getType(),
                                         getMask().getType());
}

// Per-pair value/mask shape parity. Counts pinned by SameVariadicOperandSize;
// element-type parity on parent hc.generic.
LogicalResult HCYieldPredicatedOp::verify() {
  for (auto [value, mask] : llvm::zip_equal(getValues(), getMasks())) {
    if (failed(checkPredicateShapeMatchesValue(getOperation(), value.getType(),
                                               mask.getType())))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SymbolUser verify for call ops: callee existence + kind always; signature
// parity when callee has function_type. !hc.undef escapes.
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

// EffectClass -> MemoryEffects on default resource. Pure=none;
// absent->Read+Write.
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
  // const_kwarg whitelist enforced; extra call-site attrs allowed.
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
// hc.generic parse/print/verify. Custom format; iter/ins/outs interleave
// SSA, attrs, types. Shape:
//   hc.generic
//       iter (parallel i = %m : index, reduction k = %kn : index)
//       ins  (%a at #hc.expr<"i*K+k"> : !hc.bare_tensor<...>, ...)
//       outs (%c at #hc.expr<"i*N+j"> : !hc.bare_tensor<...>)
//       -> (!hc.bare_tensor<...>, ...) { ^bb0(%av: f16, ...): ... }
// iter and outs >=1; ins may be empty.
//===----------------------------------------------------------------------===//

namespace {

// Body-arg element from operand:
//   shaped HC (incl. !hc.buffer) -> element type
//   !hc.ptr with pointee         -> pointee
//   opaque ptr / !hc.undef       -> null (no parity check)
static Type genericOperandElement(Type type) {
  if (isHCUndefType(type))
    return {};
  if (auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(type))
    return shaped.getSymbolicElementType();
  if (auto ptr = llvm::dyn_cast<PtrType>(type))
    return ptr.getElementType();
  return {};
}

// Memory carriers (ptr/buffer) bypass hc.generic SSA-result slot;
// value-semantic outs produce results in declaration order. !hc.undef counts as
// value-typed.
static bool isMemoryCarrierOperand(Type type) {
  return llvm::isa<PtrType, BufferType>(type);
}

// `<kind> <name> = %bound : type` iter entry; kind in {parallel, reduction}.
static ParseResult parseGenericIterEntry(
    OpAsmParser &parser, MLIRContext *ctx, SmallVectorImpl<Attribute> &iterSyms,
    SmallVectorImpl<Attribute> &iterKinds,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &boundsOps,
    SmallVectorImpl<Type> &boundsTypes) {
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
}

static ParseResult parseGenericIterClause(
    OpAsmParser &parser, SmallVectorImpl<Attribute> &iterSyms,
    SmallVectorImpl<Attribute> &iterKinds,
    SmallVectorImpl<OpAsmParser::UnresolvedOperand> &boundsOps,
    SmallVectorImpl<Type> &boundsTypes) {
  if (parser.parseKeyword("iter") || parser.parseLParen())
    return failure();
  // Empty `iter ()` rejected here for tighter diagnostic location.
  MLIRContext *ctx = parser.getContext();
  if (parseGenericIterEntry(parser, ctx, iterSyms, iterKinds, boundsOps,
                            boundsTypes))
    return failure();
  while (succeeded(parser.parseOptionalComma()))
    if (parseGenericIterEntry(parser, ctx, iterSyms, iterKinds, boundsOps,
                              boundsTypes))
      return failure();
  return parser.parseRParen();
}

// Single `#hc.expr<...>` axis attr.
static ParseResult
parseGenericOperandAxisAttr(OpAsmParser &parser,
                            SmallVectorImpl<Attribute> &axisOffsets) {
  Attribute axisAttr;
  auto attrLoc = parser.getCurrentLocation();
  if (parser.parseAttribute(axisAttr))
    return failure();
  auto expr = llvm::dyn_cast<ExprAttr>(axisAttr);
  if (!expr)
    return parser.emitError(attrLoc) << "expected #hc.expr<...> attribute";
  axisOffsets.push_back(expr);
  return success();
}

// `[a, b, c]` offset list; brackets shield trailing `: type` from
// parseExtendedAttr.
static ParseResult
parseGenericOperandOffsetList(OpAsmParser &parser,
                              SmallVectorImpl<Attribute> &axisOffsets) {
  if (succeeded(parser.parseOptionalRSquare()))
    return success();
  if (parseGenericOperandAxisAttr(parser, axisOffsets))
    return failure();
  while (succeeded(parser.parseOptionalComma()))
    if (parseGenericOperandAxisAttr(parser, axisOffsets))
      return failure();
  return parser.parseRSquare();
}

// `%val at [offsets] : type` operand entry.
static ParseResult
parseGenericOperandEntry(OpAsmParser &parser, MLIRContext *ctx,
                         SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ops,
                         SmallVectorImpl<Type> &types,
                         SmallVectorImpl<Attribute> &offsets) {
  OpAsmParser::UnresolvedOperand op;
  Type ty;
  if (parser.parseOperand(op) || parser.parseKeyword("at") ||
      parser.parseLSquare())
    return failure();
  // Empty offsets accepted; verifier emits rank-mismatch.
  SmallVector<Attribute> axisOffsets;
  if (parseGenericOperandOffsetList(parser, axisOffsets))
    return failure();
  if (parser.parseColonType(ty))
    return failure();
  ops.push_back(op);
  offsets.push_back(ArrayAttr::get(ctx, axisOffsets));
  types.push_back(ty);
  return success();
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
  if (parseGenericOperandEntry(parser, ctx, ops, types, offsets))
    return failure();
  while (succeeded(parser.parseOptionalComma()))
    if (parseGenericOperandEntry(parser, ctx, ops, types, offsets))
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
        // perOperand is dependent; `.template` reaches member template.
        llvm::interleaveComma(perOperand.template getAsRange<ExprAttr>(), p,
                              [&](ExprAttr e) { p.printAttribute(e); });
        p << "] : " << val.getType();
      });
  p << ")";
}

} // namespace

// `ambient (%a as "name" : type, ...)`. Operand !hc.idx or index. Empty ->
// ambient-context resolution. parseString preserves trailing `: !hc.idx<...>`.
static ParseResult parseGenericAmbientEntry(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ops,
    SmallVectorImpl<Type> &types, SmallVectorImpl<Attribute> &syms) {
  OpAsmParser::UnresolvedOperand op;
  std::string symName;
  Type ty;
  if (parser.parseOperand(op) || parser.parseKeyword("as") ||
      parser.parseString(&symName) || parser.parseColonType(ty))
    return failure();
  ops.push_back(op);
  types.push_back(ty);
  syms.push_back(parser.getBuilder().getStringAttr(symName));
  return success();
}

// Parser-time uniqueness; verifier re-checks later.
static ParseResult validateAmbientSymUniqueness(OpAsmParser &parser,
                                                ArrayRef<Attribute> syms) {
  llvm::StringSet<> seen;
  for (Attribute a : syms) {
    auto s = cast<StringAttr>(a).getValue();
    if (s.empty())
      return parser.emitError(parser.getCurrentLocation())
             << "ambient sym name must be non-empty";
    if (!seen.insert(s).second)
      return parser.emitError(parser.getCurrentLocation())
             << "duplicate ambient sym '" << s << "'";
  }
  return success();
}

static ParseResult parseGenericAmbientClause(
    OpAsmParser &parser, SmallVectorImpl<OpAsmParser::UnresolvedOperand> &ops,
    SmallVectorImpl<Type> &types, SmallVectorImpl<Attribute> &syms) {
  if (failed(parser.parseOptionalKeyword("ambient")))
    return success();
  if (parser.parseLParen())
    return failure();
  if (succeeded(parser.parseOptionalRParen()))
    return success();
  if (parseGenericAmbientEntry(parser, ops, types, syms))
    return failure();
  while (succeeded(parser.parseOptionalComma()))
    if (parseGenericAmbientEntry(parser, ops, types, syms))
      return failure();
  if (parser.parseRParen())
    return failure();
  return validateAmbientSymUniqueness(parser, syms);
}

// Per-clause state shared between HCGenericOp::parse phases.
namespace {
struct HCGenericParseState {
  SmallVector<Attribute> iterSyms;
  SmallVector<Attribute> iterKinds;
  SmallVector<OpAsmParser::UnresolvedOperand> boundsOps;
  SmallVector<Type> boundsTypes;
  SmallVector<OpAsmParser::UnresolvedOperand> insOps;
  SmallVector<Type> insTypes;
  SmallVector<Attribute> insOffsets;
  SmallVector<OpAsmParser::UnresolvedOperand> outsOps;
  SmallVector<Type> outsTypes;
  SmallVector<Attribute> outsOffsets;
  SmallVector<OpAsmParser::UnresolvedOperand> ambientOps;
  SmallVector<Type> ambientTypes;
  SmallVector<Attribute> ambientSyms;
  SmallVector<Type> resultTypes;
};
} // namespace

// Parse iter/ins/outs/ambient into state.
static ParseResult parseHCGenericClauses(OpAsmParser &parser,
                                         HCGenericParseState &state) {
  if (parseGenericIterClause(parser, state.iterSyms, state.iterKinds,
                             state.boundsOps, state.boundsTypes))
    return failure();
  if (parseGenericOperandClause(parser, "ins", /*allowEmpty=*/true,
                                state.insOps, state.insTypes, state.insOffsets))
    return failure();
  if (parseGenericOperandClause(parser, "outs", /*allowEmpty=*/false,
                                state.outsOps, state.outsTypes,
                                state.outsOffsets))
    return failure();
  return parseGenericAmbientClause(parser, state.ambientOps, state.ambientTypes,
                                   state.ambientSyms);
}

// Trailing `-> (T0, T1, ...)`; empty allowed.
static ParseResult parseHCGenericResultArrow(OpAsmParser &parser,
                                             SmallVectorImpl<Type> &outTypes) {
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (succeeded(parser.parseOptionalRParen()))
    return success();
  if (parser.parseTypeList(outTypes))
    return failure();
  return parser.parseRParen();
}

// Resolve in ODS order: iter_bounds, ins, outs, ambient_idxs.
static ParseResult resolveHCGenericOperands(OpAsmParser &parser,
                                            const HCGenericParseState &state,
                                            OperationState &result) {
  auto loc = parser.getCurrentLocation();
  if (parser.resolveOperands(state.boundsOps, state.boundsTypes, loc,
                             result.operands) ||
      parser.resolveOperands(state.insOps, state.insTypes, loc,
                             result.operands) ||
      parser.resolveOperands(state.outsOps, state.outsTypes, loc,
                             result.operands) ||
      parser.resolveOperands(state.ambientOps, state.ambientTypes, loc,
                             result.operands))
    return failure();
  return success();
}

static void populateHCGenericAttrs(OpAsmParser &parser,
                                   const HCGenericParseState &state,
                                   OperationState &result) {
  MLIRContext *ctx = parser.getContext();
  auto &builder = parser.getBuilder();
  result.addAttribute(HCGenericOp::getIterSymsAttrName(result.name),
                      ArrayAttr::get(ctx, state.iterSyms));
  result.addAttribute(HCGenericOp::getIterKindsAttrName(result.name),
                      ArrayAttr::get(ctx, state.iterKinds));
  result.addAttribute(HCGenericOp::getAmbientIdxSymsAttrName(result.name),
                      ArrayAttr::get(ctx, state.ambientSyms));
  result.addAttribute(HCGenericOp::getInsOffsetsAttrName(result.name),
                      ArrayAttr::get(ctx, state.insOffsets));
  result.addAttribute(HCGenericOp::getOutsOffsetsAttrName(result.name),
                      ArrayAttr::get(ctx, state.outsOffsets));
  result.addAttribute(HCGenericOp::getOperandSegmentSizesAttrName(result.name),
                      builder.getDenseI32ArrayAttr(
                          {static_cast<int32_t>(state.boundsOps.size()),
                           static_cast<int32_t>(state.insOps.size()),
                           static_cast<int32_t>(state.outsOps.size()),
                           static_cast<int32_t>(state.ambientOps.size())}));
}

ParseResult HCGenericOp::parse(OpAsmParser &parser, OperationState &result) {
  HCGenericParseState state;
  if (parseHCGenericClauses(parser, state) ||
      parseHCGenericResultArrow(parser, state.resultTypes) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{},
                         /*enableNameShadowing=*/false))
    return failure();

  if (resolveHCGenericOperands(parser, state, result))
    return failure();

  populateHCGenericAttrs(parser, state, result);
  result.addTypes(state.resultTypes);
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

  // Ambient sym clause prints only when populated.
  OperandRange ambient = getAmbientIdxs();
  if (!ambient.empty()) {
    p << " ambient (";
    llvm::interleaveComma(
        llvm::zip_equal(ambient,
                        getAmbientIdxSymsAttr().getAsRange<StringAttr>()),
        p, [&](auto pair) {
          auto [val, sym] = pair;
          p << val << " as " << sym << " : " << val.getType();
        });
    p << ")";
  }

  p << " -> (";
  llvm::interleaveComma(getResultTypes(), p, [&](Type t) { p.printType(t); });
  p << ")";

  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{getIterSymsAttrName(), getIterKindsAttrName(),
                       getInsOffsetsAttrName(), getOutsOffsetsAttrName(),
                       getAmbientIdxSymsAttrName(),
                       getOperandSegmentSizesAttrName()});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true);
}

// iter clause: triple-arity, non-empty, unique non-empty names.
// Populates reductionSyms for output-offset check.
static LogicalResult
verifyHCGenericIterClause(HCGenericOp op,
                          llvm::SmallDenseSet<StringRef> &reductionSyms) {
  ArrayAttr iterSyms = op.getIterSymsAttr();
  ArrayAttr iterKinds = op.getIterKindsAttr();
  OperandRange iterBounds = op.getIterBounds();
  if (iterSyms.size() != iterBounds.size() ||
      iterSyms.size() != iterKinds.size())
    return op.emitOpError(
               "iter_syms, iter_bounds, and iter_kinds must agree on "
               "iter count, got ")
           << iterSyms.size() << ", " << iterBounds.size() << ", "
           << iterKinds.size();
  if (iterSyms.empty())
    return op.emitOpError("must declare at least one iter");

  llvm::StringSet<> seenSyms;
  for (auto [symAttr, kindAttr] :
       llvm::zip_equal(iterSyms.getAsRange<StringAttr>(),
                       iterKinds.getAsRange<IterKindAttr>())) {
    StringRef name = symAttr.getValue();
    if (name.empty())
      return op.emitOpError("iter sym names must be non-empty");
    if (!seenSyms.insert(name).second)
      return op.emitOpError("duplicate iter sym '") << name << "'";
    if (kindAttr.getValue() == IterKind::Reduction)
      reductionSyms.insert(name);
  }
  return success();
}

// True iff expr is the bare leaf reference to single symbol `name`.
static bool exprIsBareSymbol(sym::ExprHandle expr, StringRef name) {
  StringRef onlyName;
  bool unique = true;
  sym::walkSymbolNames(expr, [&](StringRef n) {
    if (onlyName.empty())
      onlyName = n;
    else if (onlyName != n)
      unique = false;
  });
  return unique && !onlyName.empty() && onlyName == name;
}

// Single ambient pair: non-empty unique name; if operand is !hc.idx<sym>
// the expression must pin the same bare sym. Post strip to index -> vacuous.
static LogicalResult verifyHCGenericAmbientEntry(HCGenericOp op, Value val,
                                                 StringRef name,
                                                 llvm::StringSet<> &seen) {
  if (name.empty())
    return op.emitOpError("ambient sym name must be non-empty");
  if (!seen.insert(name).second)
    return op.emitOpError("duplicate ambient sym '") << name << "'";
  auto idxTy = dyn_cast<IdxType>(val.getType());
  if (!idxTy)
    return success();
  ExprAttr expr = idxTy.getExpr();
  if (!expr)
    return success();
  if (!exprIsBareSymbol(expr.getValue(), name))
    return op.emitOpError("ambient sym '")
           << name << "' operand type does not pin the same bare symbol";
  return success();
}

// Count parity + per-entry check via verifyHCGenericAmbientEntry.
static LogicalResult verifyHCGenericAmbientClause(HCGenericOp op) {
  OperandRange ambientIdxs = op.getAmbientIdxs();
  ArrayAttr ambientSyms = op.getAmbientIdxSymsAttr();
  if (ambientIdxs.size() != ambientSyms.size())
    return op.emitOpError("ambient_idxs count ")
           << ambientIdxs.size() << " != ambient_idx_syms count "
           << ambientSyms.size();
  llvm::StringSet<> seen;
  for (auto [val, symAttr] :
       llvm::zip_equal(ambientIdxs, ambientSyms.getAsRange<StringAttr>()))
    if (failed(verifyHCGenericAmbientEntry(op, val, symAttr.getValue(), seen)))
      return failure();
  return success();
}

// One SSA result per value-typed out; ptr/buffer outs land in memory only.
static LogicalResult verifyHCGenericResultsVsOuts(HCGenericOp op) {
  if (op.getOuts().empty())
    return op.emitOpError("must declare at least one output");
  SmallVector<Value> valueOuts;
  for (Value out : op.getOuts())
    if (!isMemoryCarrierOperand(out.getType()))
      valueOuts.push_back(out);
  if (op.getResults().size() != valueOuts.size())
    return op.emitOpError("results count ")
           << op.getResults().size() << " != value-typed outs count "
           << valueOuts.size() << " (ptr/buffer outs contribute no SSA result)";
  for (auto [i, resTy, outVal] :
       llvm::enumerate(op.getResultTypes(), valueOuts))
    if (resTy != outVal.getType())
      return op.emitOpError("result #")
             << i << " type " << resTy << " does not match outs operand type "
             << outVal.getType();
  return success();
}

// !hc.undef: nullopt (rank deferred). !hc.ptr: rank 1 (single linear addr).
static std::optional<size_t> hcGenericOperandRank(Type t) {
  if (isHCUndefType(t))
    return std::nullopt;
  if (llvm::isa<PtrType>(t))
    return size_t{1};
  if (auto shaped = llvm::dyn_cast<SymbolicallyShapedTypeInterface>(t))
    if (ShapeAttr shape = shaped.getSymbolicShape())
      return shape.getDims().size();
  return std::nullopt;
}

// Per-operand offset array length == operand rank. Pre-flatten: logical
// rank; post-flatten: 1D on both sides.
static LogicalResult verifyHCGenericPerOperandOffsetShape(HCGenericOp op,
                                                          ArrayAttr offsets,
                                                          OperandRange operands,
                                                          StringRef role) {
  if (offsets.size() != operands.size())
    return op.emitOpError(role)
           << "_offsets count " << offsets.size() << " != " << role << " count "
           << operands.size();
  for (auto [i, perOperandOffsets, operand] :
       llvm::enumerate(offsets.getAsRange<ArrayAttr>(), operands)) {
    auto entries = llvm::dyn_cast<ArrayAttr>(perOperandOffsets);
    if (!entries)
      return op.emitOpError(role)
             << "_offsets[" << i
             << "] must be an array of #hc.expr<...> per axis";
    if (auto rank = hcGenericOperandRank(operand.getType()))
      if (entries.size() != *rank)
        return op.emitOpError(role)
               << "_offsets[" << i << "] has " << entries.size()
               << " axis expression(s), expected " << *rank
               << " (one per operand axis)";
  }
  return success();
}

// Reduction iters in output offsets would write the same slot multiple
// times without a combinator; not modeled. Reduction value rides on
// outs-as-init body-arg/yield channel instead.
static LogicalResult verifyHCGenericReductionsNotInOutOffsets(
    HCGenericOp op, ArrayAttr outsOffsets,
    const llvm::SmallDenseSet<StringRef> &reductionSyms) {
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
      return op.emitOpError("output #")
             << i << " axis " << badAxis
             << " offset references reduction iter '" << bad
             << "'; only parallel iters may appear in output addressing";
  }
  return success();
}

// `!hc.pred` is the dialect's i1; `i1` itself is its post-launch-body
// substrate. Body block arg often outlives a `bare_tensor<!hc.pred>` ->
// `ptr<workgroup, i1>` operand swap; treat the two as interchangeable here.
static bool isPredAndI1(Type a, Type b) {
  return (isa<PredType>(a) && b.isInteger(1)) ||
         (isa<PredType>(b) && a.isInteger(1));
}

// Body block exists, arg arity = ins+outs, each block arg type matches the
// corresponding operand element (`!hc.undef` escapes).
static LogicalResult verifyHCGenericBodyBlockArgs(HCGenericOp op,
                                                  Block &entry) {
  size_t expectedArgs = op.getIns().size() + op.getOuts().size();
  if (entry.getNumArguments() != expectedArgs)
    return op.emitOpError("body block takes ")
           << entry.getNumArguments() << " argument(s), expected "
           << expectedArgs << " (one per ins/outs)";
  auto checkArg = [&](size_t blockIdx, Value operand, StringRef role,
                      size_t roleIdx) -> LogicalResult {
    Type elem = genericOperandElement(operand.getType());
    if (!elem)
      return success();
    Type blockArgType = entry.getArgument(blockIdx).getType();
    if (isHCUndefType(blockArgType) || blockArgType == elem ||
        isPredAndI1(blockArgType, elem))
      return success();
    return op.emitOpError("body argument #")
           << blockIdx << " type " << blockArgType << " does not match " << role
           << " #" << roleIdx << " element type " << elem;
  };
  for (auto [i, in] : llvm::enumerate(op.getIns()))
    if (failed(checkArg(i, in, "ins", i)))
      return failure();
  for (auto [i, out] : llvm::enumerate(op.getOuts()))
    if (failed(checkArg(op.getIns().size() + i, out, "outs", i)))
      return failure();
  return success();
}

// Body terminator: `hc.yield` or `hc.yield_predicated`. Same type-parity
// rules; per-pair mask shape check is on the predicated op itself.
static LogicalResult verifyHCGenericBodyTerminator(HCGenericOp op,
                                                   Block &entry) {
  Operation *terminator = tryGetTerminator(entry);
  ValueRange yieldValues;
  StringRef terminatorName;
  if (auto yield = llvm::dyn_cast_or_null<HCYieldOp>(terminator)) {
    yieldValues = yield.getValues();
    terminatorName = "hc.yield";
  } else if (auto yieldPred =
                 llvm::dyn_cast_or_null<HCYieldPredicatedOp>(terminator)) {
    yieldValues = yieldPred.getValues();
    terminatorName = "hc.yield_predicated";
  } else {
    return op.emitOpError(
        "body must terminate with `hc.yield` or `hc.yield_predicated`");
  }
  if (yieldValues.size() != op.getOuts().size())
    return op.emitOpError(terminatorName)
           << " arity " << yieldValues.size() << " != outs count "
           << op.getOuts().size();
  for (auto [i, yv, outVal] : llvm::enumerate(yieldValues, op.getOuts())) {
    Type yieldType = yv.getType();
    Type outElem = genericOperandElement(outVal.getType());
    if (!outElem || isHCUndefType(yieldType) || yieldType == outElem)
      continue;
    return op.emitOpError(terminatorName)
           << " #" << i << " type " << yieldType << " does not match outs #"
           << i << " element type " << outElem;
  }
  return success();
}

LogicalResult HCGenericOp::verify() {
  llvm::SmallDenseSet<StringRef> reductionSyms;
  if (failed(verifyHCGenericIterClause(*this, reductionSyms)))
    return failure();

  ArrayAttr insOffsets = getInsOffsetsAttr();
  ArrayAttr outsOffsets = getOutsOffsetsAttr();
  if (failed(verifyHCGenericPerOperandOffsetShape(*this, insOffsets, getIns(),
                                                  "ins")) ||
      failed(verifyHCGenericPerOperandOffsetShape(*this, outsOffsets, getOuts(),
                                                  "outs")))
    return failure();

  if (failed(verifyHCGenericAmbientClause(*this)) ||
      failed(verifyHCGenericResultsVsOuts(*this)) ||
      failed(verifyHCGenericReductionsNotInOutOffsets(*this, outsOffsets,
                                                      reductionSyms)))
    return failure();

  Region &body = getBody();
  if (body.empty())
    return emitOpError("expected a body region with an entry block");
  Block &entry = body.front();
  if (failed(verifyHCGenericBodyBlockArgs(*this, entry)) ||
      failed(verifyHCGenericBodyTerminator(*this, entry)))
    return failure();
  return success();
}

// Per-operand effects: ptr/buffer `ins` Read; ptr/buffer `outs` Read+Write
// (read sources the carry on a memory destination). Value-typed slots: none.
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
