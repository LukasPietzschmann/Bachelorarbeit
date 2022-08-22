#include "code_gen.h"

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Verifier.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "plattform.h"

#define assert_not_reached() assert(0 && "should not reach")

using namespace llvm;
struct ValueContext {
	using Ptr = std::shared_ptr<ValueContext>;

	struct ValueOrArgIndex {
		explicit ValueOrArgIndex(Value* value, AllocaInst* outside_alloca = nullptr) :
			outside_alloca(outside_alloca), value(value), index(-1) {}
		explicit ValueOrArgIndex(int index, AllocaInst* outside_alloca = nullptr) :
			outside_alloca(outside_alloca), value(nullptr), index(index) {}

		// index wird später benötigt, um value auf den richtigen parameter aus der signatur zu setzen.
		// outside_alloca zeigt auf die Instruktion, die das argument für den Parameter erstellt hat (die Instruktion
		// steht außerhalb der erzeugten Funktion)
		AllocaInst* outside_alloca;
		Value* value;
		int index;
	};

	explicit ValueContext(ValueContext::Ptr parent = nullptr) : parent(parent) {}
	static Ptr mk_ptr(ValueContext::Ptr parent = nullptr) { return std::make_shared<ValueContext>(parent); }

	std::map<std::string, ValueOrArgIndex> named_values;
	ValueContext::Ptr parent;

	ValueOrArgIndex* lookup_value(const std::string& name) {
		if(const auto& it = named_values.find(name); it != named_values.end())
			return &it->second;
		if(parent != nullptr)
			return parent->lookup_value(name);
		return nullptr;
	}
};

ValueContext::Ptr current_context;
Module* llvm_module;
IRBuilder<>* ir_builder;
int settings_flags;

namespace utils {
// Auch wenn LLVM selber durch das Anhängen von Zahlen dafür sorgt, dass Namen immer eindeutig sind
// ist mir aufgefallen, dass diese oft nicht aufsteigend, sondern durcheinander sind.
// Das hat mich gestört ':) Also habe ich eine eigene Funktion dafür implementiert
std::string mkuniq(const std::string& name) {
	static std::map<std::string, int> already_seen_names;
	std::stringstream ss;
	if(name.empty())
		return "";
	int& index = already_seen_names[name];
	ss << name << "." << index++;
	return ss.str();
}

Constant* construct_string(const std::string& string, const std::string& usage_hint = "") {
	// Hier sollte es okay sein einen Cache zu verwenden, da MOSTflexiPL selbst keine Strings kennt
	// und die Strings die dieses Backend erzeugt immer konstant verwendet werden
	static std::map<std::string, Constant*> string_cache;
	if(const auto& it = string_cache.find(string); it != string_cache.end())
		return it->second;

	Constant* string_constant = ConstantDataArray::getString(llvm_module->getContext(), string);
	Constant* global_string_variable = new GlobalVariable(*llvm_module,
			string_constant->getType(),
			true,
			GlobalValue::PrivateLinkage,
			string_constant,
			usage_hint);
	Constant* zero_constant = Constant::getNullValue(IntegerType::get(llvm_module->getContext(), INTEGER_WIDTH));
	Constant* string_ptr = ConstantExpr::getGetElementPtr(string_constant->getType(),
			global_string_variable,
			(Constant* [2]){zero_constant, zero_constant},
			true);
	string_cache.try_emplace(string, string_ptr);
	return string_ptr;
}

IntegerType* get_integer_type(int width = INTEGER_WIDTH) { return IntegerType::get(llvm_module->getContext(), width); }

PointerType* get_ptr_type(llvm::Type* underlying_type, int indirections = 1) {
	assert(indirections > 0);
	assert(underlying_type);
	if(indirections == 1)
		return PointerType::get(underlying_type, DEFAULT_ADDRESS_SPACE);
	return PointerType::get(get_ptr_type(underlying_type, indirections - 1), DEFAULT_ADDRESS_SPACE);
}

PointerType* get_integer_ptr_type(int width = INTEGER_WIDTH, int indirections = 1) {
	return get_ptr_type(get_integer_type(width), indirections);
}

ConstantInt* get_integer_constant(int64_t constant, int width = INTEGER_WIDTH, bool is_signed = true) {
	return ConstantInt::get(llvm_module->getContext(), APInt(width, constant, is_signed));
}

template <typename Lambda>
void generate_on_basic_block(BasicBlock* basic_block, Lambda lambda, bool reset_insert_point = false) {
	BasicBlock* prev_insert_block = ir_builder->GetInsertBlock();
	ir_builder->SetInsertPoint(basic_block);
	lambda();
	if(reset_insert_point)
		ir_builder->SetInsertPoint(prev_insert_block);
}

template <typename... Args>
Value* extract_from_v(Value* source, const std::string& hint, Args... indices) {
	std::vector<Value*> gep_args = {indices...};
	Value* adr_to_element = ir_builder->CreateGEP(source->getType()->getPointerElementType(),
			source,
			gep_args,
			hint.empty() ? "" : hint + ".ptr");
	return ir_builder->CreateLoad(adr_to_element->getType()->getPointerElementType(), adr_to_element, hint);
}

template <typename... Args>
Value* extract_from_i(Value* source, const std::string& hint, Args... indices) {
	return extract_from_v(source, hint, utils::get_integer_constant(indices)...);
}

template <typename... Args>
void insert_into(Value* destination,
		llvm::Type* type_to_gep_over,
		Value* value,
		const std::string name,
		Args... indices) {
	std::vector<Value*> gep_args = {utils::get_integer_constant(indices)...};
	Value* adr_to_element = ir_builder->CreateGEP(type_to_gep_over, destination, gep_args, name);
	ir_builder->CreateStore(value, adr_to_element);
}

template <typename... Args>
void insert_into(Value* destination, Value* value, const std::string name, Args... indices) {
	insert_into(destination, destination->getType()->getPointerElementType(), value, name, indices...);
}

template <typename... Args>
void insert_into(Value* destination, int value, const std::string name, Args... indices) {
	insert_into(destination, get_integer_constant(value), name, indices...);
}

template <typename... Args>
void insert_into(Value* destination, bool value, const std::string name, Args... indices) {
	insert_into(destination, get_integer_constant(value, BOOL_WIDTH, false), name, indices...);
}
}
std::string operator"" _uniq(const char* str, size_t size) { return utils::mkuniq(str); }

namespace predef_globals {
Value* unique();
}
namespace flx_value {
enum Flags { nat = 1 << 0, synt = 1 << 1, nil = 1 << 2 };

Value* is(Value* value, Flags flag) {
	assert(value->getType()->isIntegerTy(FLX_VALUE_WIDTH));
	Value* mask_result = ir_builder->CreateAnd(value, static_cast<int64_t>(flag) << FLX_BASE_WIDTH);
	std::string name = "value.is.";
	switch(flag) {
		case nil: name += "nil"; break;
		case nat: name += "nat"; break;
		case synt: name += "synt"; break;
	}
	return ir_builder->CreateICmpNE(mask_result, utils::get_integer_constant(0, FLX_VALUE_WIDTH), utils::mkuniq(name));
}

ConstantInt* get(Flags flag, int32_t value = 0) {
	// Da nil als false interpretiert wird, muss der Wert 0 sein
	if(flag == nil)
		assert(value == 0);
	if(flag == synt)
		assert(value != 0);
	int64_t result = (static_cast<int64_t>(flag) << FLX_BASE_WIDTH) + value;
	return utils::get_integer_constant(result, FLX_VALUE_WIDTH, false);
}

// Wenn strict gesetzt ist, dann muss value bool oder int sein
// Andernfalls ist es auch in Ordnung, wenn value eine flx_value ist.
// Aktuell wird strict nur von bin_gen nicht gesetzt
Value* promote_to_flx_value(Value* value, Flags flag, bool strict = true) {
	if(!strict && value->getType()->isIntegerTy(FLX_VALUE_WIDTH))
		return value;
	assert(value->getType()->isIntegerTy(INTEGER_WIDTH) || value->getType()->isIntegerTy(BOOL_WIDTH));
	// Da nil als false interpretiert wird, muss der Wert 0 sein.
	// Dies kann zur Übersetzungszeit allerdings nur für Konstanten überprüft werden
	if(flag == nil && isa<ConstantInt>(value))
		assert(cast<ConstantInt>(value)->isZero());
	Value* extended = ir_builder->CreateZExt(value, utils::get_integer_type(FLX_VALUE_WIDTH));
	return ir_builder->CreateAdd(extended,
			utils::get_integer_constant(static_cast<int64_t>(flag) << FLX_BASE_WIDTH, FLX_VALUE_WIDTH));
}

Value* truncate_to_raw_value(Value* value, int width = INTEGER_WIDTH) {
	assert(value->getType()->isIntegerTy(FLX_VALUE_WIDTH));
	return ir_builder->CreateTrunc(value, utils::get_integer_type(width));
}

Value* exec_if(std::function<Value*()> check, std::function<Value*()> check_true, std::function<Value*()> check_false) {
	BasicBlock* check_bb = BasicBlock::Create(llvm_module->getContext(), "check"_uniq);
	BasicBlock* check_true_bb = BasicBlock::Create(llvm_module->getContext(), "check.true"_uniq);
	BasicBlock* merge_bb = BasicBlock::Create(llvm_module->getContext(), "check.merge"_uniq);

	Value* check_false_result = check_false();

	ir_builder->CreateBr(check_bb);
	ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(check_bb);
	utils::generate_on_basic_block(check_bb, [&]() { ir_builder->CreateCondBr(check(), check_true_bb, merge_bb); });

	Value* result;
	ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(check_true_bb);
	utils::generate_on_basic_block(check_true_bb, [&]() {
		result = check_true();
		ir_builder->CreateBr(merge_bb);
	});

	PHINode* phi;
	ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(merge_bb);
	utils::generate_on_basic_block(merge_bb, [&]() {
		phi = ir_builder->CreatePHI(utils::get_integer_type(FLX_VALUE_WIDTH), 2);
		phi->addIncoming(check_false_result, check_bb);
		phi->addIncoming(result, check_true_bb);
	});

	return phi;
}

Value* unique_synt() {
	Value* unique =
			ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), predef_globals::unique(), "synt"_uniq);
	Value* next_unique =
			ir_builder->CreateAdd(flx_value::truncate_to_raw_value(unique), utils::get_integer_constant(1));
	ir_builder->CreateStore(flx_value::promote_to_flx_value(next_unique, flx_value::synt), predef_globals::unique());
	return unique;
}
}

namespace predef_function_names {
static const std::string& main = "main";
static const std::string& print = "printf";
static const std::string& sprint = "sprintf";
static const std::string& scan = "scanf";
static const std::string& pow = "pow";
static const std::string& fac = "fac";
static const std::string& init_map = "init_map";
static const std::string& destroy_map = "destroy_map";
static const std::string& search = "search";
static const std::string& enter = "enter";
static const std::string& malloc = "malloc";
static const std::string& strcat = "strcat";
}

namespace predef_structure_names {
static const std::string& bracket = "bracket";
static const std::string& hsearch_entry = "entry";
}

namespace predef_global_names {
static const std::string& nil = "nil";
static const std::string& trueval = "trueval";
static const std::string& unique = "next.unique";
}

namespace predef_structures {
StructType* bracket() {
	static StructType* bracket = nullptr;
	if(bracket != nullptr)
		return bracket;

	bracket = StructType::create(llvm_module->getContext(), predef_structure_names::bracket);
	// i32*: Gewählte Option pro Durchlauf (Alle)
	// i32: Anzahl der Durchläufe (Wiederholung)
	// i1: Wurde eine Option gewählt (Option)
	// bracket***: Liste an Klammern aus jedem Durchläufe (Alle)
	std::vector<llvm::Type*> bracket_elements = {utils::get_ptr_type(utils::get_integer_type()),
			utils::get_integer_type(),
			utils::get_integer_type(BOOL_WIDTH),
			utils::get_ptr_type(bracket, 3)};
	bracket->setBody(bracket_elements);

	return bracket;
}

StructType* hsearch_entry() {
	static StructType* hsearch_entry = nullptr;
	if(hsearch_entry != nullptr)
		return hsearch_entry;

	hsearch_entry = StructType::create(llvm_module->getContext(), predef_structure_names::hsearch_entry);
	// i8*: key
	// i35*: data (Eigentlich void*, aber da hier keine anderen Werte gespeichert werden, wird der Typ hier so explizit
	// angegeben)
	std::vector<llvm::Type*> bracket_elements = {utils::get_ptr_type(utils::get_integer_type(8)),
			utils::get_ptr_type(utils::get_integer_type(FLX_VALUE_WIDTH))};
	hsearch_entry->setBody(bracket_elements);

	return hsearch_entry;
}
}

namespace predef_functions {
Function* main() {
	if(Function* main_function = llvm_module->getFunction(predef_function_names::main))
		return main_function;

	FunctionType* main_function_type = FunctionType::get(utils::get_integer_type(), false);
	Function* main_function = Function::Create(main_function_type,
			Function::LinkOnceAnyLinkage,
			predef_function_names::main,
			llvm_module);
	main_function->setCallingConv(CallingConv::Fast);
	return main_function;
}

Function* printf() {
	if(Function* printf_function = llvm_module->getFunction(predef_function_names::print))
		return printf_function;

	std::vector<llvm::Type*> printf_args(1, utils::get_integer_ptr_type(CHARACTER_WIDTH));
	FunctionType* printf_function_type = FunctionType::get(utils::get_integer_type(), printf_args, true);
	Function* printf = Function::Create(printf_function_type,
			Function::ExternalLinkage,
			predef_function_names::print,
			llvm_module);
	printf->setCallingConv(CallingConv::C);
	return printf;
}

Function* sprintf() {
	if(Function* sprintf_function = llvm_module->getFunction(predef_function_names::sprint))
		return sprintf_function;

	std::vector<llvm::Type*> sprintf_args(2, utils::get_integer_ptr_type(CHARACTER_WIDTH));
	FunctionType* sprintf_function_type = FunctionType::get(utils::get_integer_type(), sprintf_args, true);
	Function* sprintf = Function::Create(sprintf_function_type,
			Function::ExternalLinkage,
			predef_function_names::sprint,
			llvm_module);
	sprintf->setCallingConv(CallingConv::C);
	return sprintf;
}

Function* scanf() {
	if(Function* scanf_function = llvm_module->getFunction(predef_function_names::scan))
		return scanf_function;

	std::vector<llvm::Type*> scanf_args(1, utils::get_integer_ptr_type(CHARACTER_WIDTH));
	FunctionType* scanf_function_type = FunctionType::get(utils::get_integer_type(), scanf_args, true);
	Function* scanf =
			Function::Create(scanf_function_type, Function::ExternalLinkage, predef_function_names::scan, llvm_module);
	scanf->setCallingConv(CallingConv::C);
	return scanf;
}

// Nur ein Wrapper um die intern implementierte llvm.powi.float.i32 Funktion
Function* pow() {
	if(Function* pow_function = llvm_module->getFunction(predef_function_names::pow))
		return pow_function;

	std::vector<llvm::Type*> pow_args(2, utils::get_integer_type(FLX_VALUE_WIDTH));
	FunctionType* pow_function_type = FunctionType::get(utils::get_integer_type(FLX_VALUE_WIDTH), pow_args, false);
	Function* pow_function =
			Function::Create(pow_function_type, Function::PrivateLinkage, predef_function_names::pow, llvm_module);

	BasicBlock* entry = BasicBlock::Create(llvm_module->getContext());
	pow_function->getBasicBlockList().push_back(entry);
	// clang-format off
	utils::generate_on_basic_block(entry, [&pow_function]() {
		std::vector<Value*> args;
		args.push_back(ir_builder->CreateCast(Instruction::SIToFP,
				flx_value::truncate_to_raw_value(pow_function->getArg(0)),
				IntegerType::getFloatTy(llvm_module->getContext())));
		args.push_back(flx_value::truncate_to_raw_value(pow_function->getArg(1)));
		std::vector<llvm::Type*> intrinsic_specialisation{IntegerType::getFloatTy(llvm_module->getContext()),
				utils::get_integer_type(32)};  // https://llvm.org/docs/LangRef.html#llvm-powi-intrinsic
		Value* result = ir_builder->CreateCall(
				Intrinsic::getDeclaration(llvm_module, Intrinsic::powi, intrinsic_specialisation),
				args);
		Value* casted_result = ir_builder->CreateCast(Instruction::FPToSI, result, utils::get_integer_type());
		ir_builder->CreateRet(flx_value::promote_to_flx_value(casted_result, flx_value::nat));
	}, true);
	// clang-format on

	pow_function->setCallingConv(CallingConv::Fast);

	verifyFunction(*pow_function, &errs());

	return pow_function;
}

Function* fac() {
	if(Function* fac_function = llvm_module->getFunction(predef_function_names::fac))
		return fac_function;

	std::vector<llvm::Type*> fac_args(1, utils::get_integer_type(FLX_VALUE_WIDTH));
	FunctionType* fac_function_type = FunctionType::get(utils::get_integer_type(FLX_VALUE_WIDTH), fac_args, false);
	Function* fac_function =
			Function::Create(fac_function_type, Function::PrivateLinkage, predef_function_names::fac, llvm_module);

	Value* operand;
	Value* multiplication_result;

	BasicBlock* entry = BasicBlock::Create(llvm_module->getContext());
	BasicBlock* loop_begin = BasicBlock::Create(llvm_module->getContext(), "loop.begin"_uniq);
	BasicBlock* loop_end = BasicBlock::Create(llvm_module->getContext(), "loop.end"_uniq);

	fac_function->getBasicBlockList().push_back(entry);
	// clang-format off
	utils::generate_on_basic_block(entry, [&]() {
		operand = flx_value::truncate_to_raw_value(fac_function->getArg(0));
		Value* br_to_while = ir_builder->CreateICmpSGT(operand, utils::get_integer_constant(1));
		ir_builder->CreateCondBr(br_to_while, loop_begin, loop_end);
	}, true);
	// clang-format on

	fac_function->getBasicBlockList().push_back(loop_begin);
	// clang-format off
	utils::generate_on_basic_block(loop_begin, [&]() {
		Value* post_decrement_result;
		PHINode* z = ir_builder->CreatePHI(utils::get_integer_type(), 2, "z"_uniq);
		PHINode* x = ir_builder->CreatePHI(utils::get_integer_type(), 2, "x"_uniq);
		post_decrement_result = ir_builder->CreateSub(x, utils::get_integer_constant(1), "x.dec"_uniq);
		multiplication_result = ir_builder->CreateMul(x, z);
		Value* branch_back_to_while = ir_builder->CreateICmpSGT(x, utils::get_integer_constant(2));
		ir_builder->CreateCondBr(branch_back_to_while, loop_begin, loop_end);
		// addIncoming darf erst aufgerufen werden, wenn alle Operanden initialisiert sind
		z->addIncoming(utils::get_integer_constant(1), entry);
		z->addIncoming(multiplication_result, loop_begin);
		x->addIncoming(post_decrement_result, loop_begin);
		x->addIncoming(operand, entry);
	}, true);
	// clang-format on

	fac_function->getBasicBlockList().push_back(loop_end);
	// clang-format off
	utils::generate_on_basic_block(loop_end, [&]() {
		PHINode* return_phi = ir_builder->CreatePHI(utils::get_integer_type(), 2);
		return_phi->addIncoming(utils::get_integer_constant(1), entry);
		return_phi->addIncoming(multiplication_result, loop_begin);
		ir_builder->CreateRet(flx_value::promote_to_flx_value(return_phi, flx_value::nat));
	}, true);
	// clang-format on

	fac_function->setCallingConv(CallingConv::Fast);

	verifyFunction(*fac_function, &errs());

	return fac_function;
}

Function* init_map() {
	if(Function* init_function = llvm_module->getFunction(predef_function_names::init_map))
		return init_function;

	FunctionType* init_function_type = FunctionType::get(llvm::Type::getVoidTy(llvm_module->getContext()), false);
	Function* init_function = Function::Create(init_function_type,
			Function::ExternalLinkage,
			predef_function_names::init_map,
			llvm_module);
	init_function->setCallingConv(CallingConv::C);
	return init_function;
}

Function* destory_map() {
	if(Function* destroy_function = llvm_module->getFunction(predef_function_names::destroy_map))
		return destroy_function;

	FunctionType* destory_function_type = FunctionType::get(llvm::Type::getVoidTy(llvm_module->getContext()), false);
	Function* destroy_function = Function::Create(destory_function_type,
			Function::ExternalLinkage,
			predef_function_names::destroy_map,
			llvm_module);
	destroy_function->setCallingConv(CallingConv::C);
	return destroy_function;
}

Function* search() {
	if(Function* search_function = llvm_module->getFunction(predef_function_names::search))
		return search_function;

	std::vector<llvm::Type*> search_args = {utils::get_integer_ptr_type(CHARACTER_WIDTH)};
	FunctionType* search_function_type =
			FunctionType::get(utils::get_ptr_type(predef_structures::hsearch_entry()), search_args, false);
	Function* search_function = Function::Create(search_function_type,
			Function::ExternalLinkage,
			predef_function_names::search,
			llvm_module);
	search_function->setCallingConv(CallingConv::C);
	return search_function;
}

Function* enter() {
	if(Function* enter_function = llvm_module->getFunction(predef_function_names::enter))
		return enter_function;

	std::vector<llvm::Type*> enter_args = {utils::get_integer_ptr_type(CHARACTER_WIDTH),
			PointerType::get(llvm_module->getContext(), DEFAULT_ADDRESS_SPACE)};
	FunctionType* enter_function_type =
			FunctionType::get(llvm::Type::getVoidTy(llvm_module->getContext()), enter_args, false);
	Function* enter_function =
			Function::Create(enter_function_type, Function::ExternalLinkage, predef_function_names::enter, llvm_module);
	enter_function->setCallingConv(CallingConv::C);
	return enter_function;
}

Function* malloc() {
	if(Function* malloc_function = llvm_module->getFunction(predef_function_names::malloc))
		return malloc_function;

	std::vector<llvm::Type*> malloc_args = {utils::get_integer_type()};
	FunctionType* malloc_function_type =
			FunctionType::get(PointerType::get(llvm_module->getContext(), DEFAULT_ADDRESS_SPACE), malloc_args, false);
	Function* malloc = Function::Create(malloc_function_type,
			Function::ExternalLinkage,
			predef_function_names::malloc,
			llvm_module);
	malloc->setCallingConv(CallingConv::C);
	return malloc;
}

Function* strcat() {
	if(Function* str_cat_function = llvm_module->getFunction(predef_function_names::strcat))
		return str_cat_function;

	std::vector<llvm::Type*> strcat_args(2, utils::get_integer_ptr_type(CHARACTER_WIDTH));
	FunctionType* str_cat_function_type =
			FunctionType::get(utils::get_integer_ptr_type(CHARACTER_WIDTH), strcat_args, false);
	Function* str_cat = Function::Create(str_cat_function_type,
			Function::ExternalLinkage,
			predef_function_names::strcat,
			llvm_module);
	str_cat->setCallingConv(CallingConv::C);
	return str_cat;
}
}

namespace predef_globals {
Value* nil() {
	if(GlobalVariable* nil = llvm_module->getNamedGlobal(predef_global_names::nil); nil != nullptr)
		return nil;
	return new GlobalVariable(*llvm_module,
			utils::get_integer_type(FLX_VALUE_WIDTH),
			true,
			GlobalValue::PrivateLinkage,
			flx_value::get(flx_value::nil),
			predef_global_names::nil);
}

Value* trueval() {
	if(GlobalVariable* trueval = llvm_module->getNamedGlobal(predef_global_names::trueval); trueval != nullptr)
		return trueval;
	return new GlobalVariable(*llvm_module,
			utils::get_integer_type(FLX_VALUE_WIDTH),
			true,
			GlobalValue::PrivateLinkage,
			flx_value::get(flx_value::synt, 1),
			predef_global_names::trueval);
}

Value* unique() {
	if(GlobalVariable* unique = llvm_module->getNamedGlobal(predef_global_names::unique); unique != nullptr)
		return unique;
	return new GlobalVariable(*llvm_module,
			utils::get_integer_type(FLX_VALUE_WIDTH),
			false,
			GlobalValue::PrivateLinkage,
			// Darf nicht bei 0 anfangen, da ein synthetischer Wert nicht als false zu interpretieren ist
			// Muss bei 2 anfangen, da die 1 von trueval verwendet wird
			flx_value::get(flx_value::synt, 2),
			predef_global_names::unique);
}
}

namespace func_utils {
enum BracketType { alternative, option, repetition };

struct AllocaRun;
struct AllocaBracket {
	AllocaBracket(CH_id_t id, int alts) : id(id), alts(alts), bracket_ptr(nullptr) {}
	CH_id_t id;
	int alts;
	Value* bracket_ptr;
	std::vector<AllocaRun> runs;
};

struct ContextRun;
struct ContextBracket {
	ContextBracket(CH_id_t id, int index, Value* bracket_ptr) :
		id(id), index_in_arg_list(index), bracket_ptr(bracket_ptr) {}
	CH_id_t id;
	// Nur gültig, falls es eine top-level Klammer ist
	int index_in_arg_list;
	Value* bracket_ptr;
	std::vector<ContextRun> runs;
	// Um in verschiedenen Durchläufen einer Klammer den aktuellen Wert eines Parameters für die aktuelle Iteration
	// abspeichern zu können, wird in diesem Kontext für jeden in der Klammer enthalten Parameter eine zusätzliche
	// Variable erstellt, die dann eben hier abgespeichert wird
	ValueContext::Ptr values_over_all_runs{nullptr};
};

struct AllocaRun {
	explicit AllocaRun(int selected_option) : selected_option(selected_option) {}
	int selected_option;
	std::vector<AllocaInst*> variables;
	std::vector<AllocaBracket> brackets;
};

struct ContextRun {
	ContextRun(const ContextRun& other) :
		selected_option(other.selected_option), ascend_in_contexts(other.ascend_in_contexts),
		// other.variabled kopieren
		variables(new ValueContext(*other.variables)), brackets(other.brackets) {}
	ContextRun(int selected_option, bool ascend_in_contexts, ValueContext::Ptr parent_context = nullptr) :
		selected_option(selected_option), ascend_in_contexts(ascend_in_contexts),
		variables(ValueContext::mk_ptr(parent_context)) {}
	int selected_option;
	// Für eckige Klammern mit nur einer Alternative gesetzt
	bool ascend_in_contexts;
	ValueContext::Ptr variables;
	std::vector<ContextBracket> brackets;
};

std::map<std::string, std::pair<bool, Expr>> available_functions_to_instantiate;
std::vector<ContextBracket>* cbrackets;

Function* instantiate_function(const std::string& static_function_name,
		std::vector<AllocaBracket>& abrackets,
		std::vector<ContextBracket>& out_cbrackets) {
	const auto& it = available_functions_to_instantiate.find(static_function_name);
	assert(it != available_functions_to_instantiate.end());
	Expr expr = it->second.second;
	bool is_static = it->second.first;

	std::vector<llvm::Type*> oper_params;
	std::vector<std::string> oper_params_names;

	const auto& new_param = [&oper_params, &oper_params_names](llvm::Type* type, const std::string& name = "") {
		oper_params.push_back(type);
		oper_params_names.push_back(name);
		return oper_params.size() - 1;
	};

	std::stringstream dynamic_function_name;
	dynamic_function_name << static_function_name;

	ValueContext::Ptr top_level_param_context = ValueContext::mk_ptr();
	std::vector<ContextBracket> cbrackets;
	AllocaBracket* current_abracket = nullptr;
	ContextBracket* current_cbracket = nullptr;
	AllocaRun* current_arun = nullptr;
	ContextRun* current_crun = nullptr;

	// Um Klammern aus verschiedenen Durchläufen die logisch dieselbe Klammer sind, aber durch verschiedene Objekte
	// repräsentiert werden denselben ValueContext Zeiger zu geben, wird hier die Klammer-ID auf den jeweiligen Zeiger
	// gemapped
	std::map<CH_id_t, ValueContext::Ptr> values_over_all_runs_per_bracket;
	const auto& setup_cbracket = [&](int& bracket_index, int index_in_arg_list, ValueContext::Ptr parent_context) {
		if(current_abracket == nullptr) {
			assert(current_arun == nullptr);
			assert(current_crun == nullptr);
			assert(abrackets.size() > bracket_index);
			if(abrackets.size() <= bracket_index)
				return false;
			current_abracket = &abrackets.at(bracket_index);
			current_cbracket =
					&cbrackets.emplace_back(current_abracket->id, index_in_arg_list, current_abracket->bracket_ptr);
			assert(cbrackets.size() == bracket_index + 1);
		} else {
			assert(current_arun != nullptr);
			assert(current_crun != nullptr);
			assert(current_arun->brackets.size() > bracket_index);
			if(current_arun->brackets.size() <= bracket_index)
				return false;
			current_abracket = &current_arun->brackets.at(bracket_index);
			current_cbracket = &current_crun->brackets.emplace_back(current_abracket->id,
					index_in_arg_list,
					current_abracket->bracket_ptr);
			assert(current_crun->brackets.size() == bracket_index + 1);
		}
		assert(current_cbracket->values_over_all_runs == nullptr);
		if(const auto& it = values_over_all_runs_per_bracket.find(current_abracket->id);
				it != values_over_all_runs_per_bracket.end()) {
			current_cbracket->values_over_all_runs = it->second;
		} else {
			ValueContext::Ptr new_voar_context = ValueContext::mk_ptr(parent_context);
			current_cbracket->values_over_all_runs = new_voar_context;
			values_over_all_runs_per_bracket.try_emplace(current_abracket->id, new_voar_context);
		}
		++bracket_index;
		return !current_abracket->runs.empty();
	};

	const auto& setup_crun = [&](int run_to_setup, ValueContext::Ptr parent_context, bool ascend_in_contexts = false) {
		current_arun = &current_abracket->runs.at(run_to_setup);
		ValueContext::Ptr parent;
		if(current_cbracket->runs.empty())
			parent = parent_context;
		else
			parent = current_cbracket->runs.at(0).variables->parent;
		current_crun = &current_cbracket->runs.emplace_back(current_arun->selected_option, ascend_in_contexts, parent);
	};

	// Hier werden alle geleseen Namen des Operators eingetragen, um mit diesen später (Im Falle eines statischen
	// Operators) einen eindeutigen Key zu erstellen
	std::stringstream names;
	enum { name, parameter, alternative, option, repetition };
	std::function<void(CH::seq<Pass>, ValueContext::Ptr)> gen_function_signatur =
			[&](CH::seq<Pass> passes, ValueContext::Ptr parent_context) {
				static int level = -1;
				++level;
				int bracket_index = 0;
				std::map<int, int> param_index_per_selected_option;
				for(Pass pass : passes) {
					AllocaBracket* old_current_abracket = current_abracket;
					ContextBracket* old_current_cbracket = current_cbracket;
					AllocaRun* old_current_arun = current_arun;
					ContextRun* old_current_crun = current_crun;

					std::stringstream parameter_name;
					switch(pass(choice_) - CH::A) {
						case name: {
							if(!is_static)
								break;
							names << pass(branch_)[CH::A](word_);
						} break;
						case parameter: {
							dynamic_function_name << "p";
							parameter_name << pass(branch_)[CH::A + 1](passes_)[CH::A](branch_)[CH::A](passes_)[CH::A](
									branch_)[CH::A](word_);
							int index = new_param(utils::get_ptr_type(utils::get_integer_type(FLX_VALUE_WIDTH)),
									utils::mkuniq(parameter_name.str() + ".ptr"));
							if(level == 0)
								top_level_param_context->named_values.try_emplace(parameter_name.str(), index);
							else {
								AllocaInst* alloca = current_arun->variables.at(
										param_index_per_selected_option[current_arun->selected_option]);
								alloca->setName(parameter_name.str() + ".ptr");
								//  An dieser Stelle existiert der Parameter, der die Variable enthält noch nicht,
								//  also wird der zugehörige Index gespeichert, der später dann durch den echten
								//  Parameter substituiert wird
								current_crun->variables->named_values.try_emplace(parameter_name.str(), index, alloca);

								if(current_cbracket->values_over_all_runs->named_values.find(parameter_name.str()) ==
										current_cbracket->values_over_all_runs->named_values.end()) {
									// Später wird der Wert mit dem Wert des Parameters aus seinem ersten Durchlauf
									// initialisiert. Deswegen wird hier der Index des Parameters aus dem ersten
									// Durchlauf abgespeichert
									current_cbracket->values_over_all_runs->named_values.try_emplace(
											parameter_name.str(),
											index);
								}
							}
							for(auto& [_, index] : param_index_per_selected_option)
								++index;
						} break;
						case alternative: {
							int index = -1;
							if(level == 0)
								index = new_param(utils::get_ptr_type(predef_structures::bracket()));

							// Eine Alternative sollte wohl immer einen Durchlauf haben
							bool has_pass = setup_cbracket(bracket_index, index, parent_context);
							if(!has_pass)
								break;

							setup_crun(0, parent_context);
							dynamic_function_name << "a" << current_abracket->runs.size()
												  << current_arun->selected_option;
							if(int selected_option = current_arun->selected_option; selected_option == 0)
								gen_function_signatur(pass(branch_)[CH::A + 2](opnd_)(row_)[CH::A](passes_),
										current_cbracket->values_over_all_runs);
							else {
								// selected_option - 1, da die 0te Option nicht in dieser Liste enthalten ist
								gen_function_signatur(pass(branch_)[CH::A + 3](passes_)[CH::A + selected_option - 1](
															  branch_)[CH::A + 1](opnd_)(row_)[CH::A](passes_),
										current_cbracket->values_over_all_runs);
							}
						} break;
						case option: {
							int index = -1;
							if(level == 0)
								index = new_param(utils::get_ptr_type(predef_structures::bracket()));

							bool has_pass = setup_cbracket(bracket_index, index, parent_context);
							if(!has_pass)
								break;

							setup_crun(0, parent_context, current_abracket->alts == 1);
							dynamic_function_name << "o" << current_abracket->runs.size()
												  << current_arun->selected_option;
							if(int selected_option = current_arun->selected_option; selected_option == 0)
								gen_function_signatur(pass(branch_)[CH::A + 2](opnd_)(row_)[CH::A](passes_),
										current_cbracket->values_over_all_runs);
							else {
								gen_function_signatur(pass(branch_)[CH::A + 3](passes_)[CH::A + selected_option - 1](
															  branch_)[CH::A + 1](opnd_)(row_)[CH::A](passes_),
										current_cbracket->values_over_all_runs);
							}
						} break;
						case repetition: {
							int index = -1;
							if(level == 0)
								index = new_param(utils::get_ptr_type(predef_structures::bracket()));

							bool has_pass = setup_cbracket(bracket_index, index, parent_context);
							if(!has_pass)
								break;

							dynamic_function_name << "r" << current_abracket->runs.size();
							for(int i = 0; i < current_abracket->runs.size(); ++i) {
								setup_crun(i, parent_context);
								dynamic_function_name << current_arun->selected_option;
								if(int selected_option = current_arun->selected_option; selected_option == 0)
									gen_function_signatur(pass(branch_)[CH::A + 2](opnd_)(row_)[CH::A](passes_),
											current_cbracket->values_over_all_runs);
								else {
									gen_function_signatur(
											pass(branch_)[CH::A + 3](passes_)[CH::A + selected_option - 1](
													branch_)[CH::A + 1](opnd_)(row_)[CH::A](passes_),
											current_cbracket->values_over_all_runs);
								}
							}
						} break;
						default: assert_not_reached();
					}
					current_abracket = old_current_abracket;
					current_cbracket = old_current_cbracket;
					current_arun = old_current_arun;
					current_crun = old_current_crun;
				}
				--level;
			};
	gen_function_signatur(expr(row_)[CH::A](passes_), current_context);

	Function* oper_function = llvm_module->getFunction(dynamic_function_name.str());
	if(oper_function == nullptr) {	// Falls die Funktion noch nicht existiert
		FunctionType* oper_function_type =
				FunctionType::get(utils::get_integer_type(FLX_VALUE_WIDTH), oper_params, false);
		oper_function = Function::Create(oper_function_type,
				Function::PrivateLinkage,
				dynamic_function_name.str(),
				llvm_module);
		oper_function->setCallingConv(CallingConv::Fast);
		for(int i = 0; i < oper_function->arg_size(); ++i)
			oper_function->getArg(i)->setName(oper_params_names.at(i));

		std::function<void(std::vector<ContextBracket>&)> substitute_indices_for_arguments_in_brackets =
				[&](std::vector<ContextBracket>& brackets) {
					for(ContextBracket& bracket : brackets) {
						for(ContextRun& run : bracket.runs) {
							substitute_indices_for_arguments_in_brackets(run.brackets);
							for(auto& [_, value_or_index] : run.variables->named_values) {
								if(value_or_index.value == nullptr)
									value_or_index.value = oper_function->getArg(value_or_index.index);
							}
						}
					}
				};
		const auto& substitute_indices_for_arguments_in_params = [&](ValueContext::Ptr top_level_param_context) {
			for(auto& [_, value_or_index] : top_level_param_context->named_values) {
				if(value_or_index.value == nullptr)
					value_or_index.value = oper_function->getArg(value_or_index.index);
			}
		};

		substitute_indices_for_arguments_in_brackets(cbrackets);
		substitute_indices_for_arguments_in_params(top_level_param_context);

		vector<ContextBracket> cbracket_copy = cbrackets;
		ValueContext::Ptr encl_run = top_level_param_context;
		ValueContext::Ptr encl_voar = top_level_param_context;
		// Variablen aus Kontexten von eckigen Klammern mit nur einer Alternative werden in ihren umschließenden
		// Kontext verschoben. Diese Änderung wird nach dem Generieren der Implementierung wieder rückgängig
		// gemacht, da sie in appl_gen für das Erzeugen der Argumente nicht von Bedeutung ist und so unnötige
		// Komplexität vermieden werden kann :)
		std::function<void(std::vector<ContextBracket>&)> ascend_contexts =
				[&](std::vector<ContextBracket>& cbrackets) {
					for(ContextBracket& bracket : cbrackets) {
						for(ContextRun& run : bracket.runs) {
							if(run.ascend_in_contexts) {
								encl_run->named_values.insert(run.variables->named_values.begin(),
										run.variables->named_values.end());
								encl_voar->named_values.insert(bracket.values_over_all_runs->named_values.begin(),
										bracket.values_over_all_runs->named_values.end());
								run.variables->named_values.clear();
							}
							encl_run = run.variables;
							encl_voar = bracket.values_over_all_runs;
							ascend_contexts(run.brackets);
						}
					}
				};
		ascend_contexts(cbrackets);

		current_context->named_values.insert(top_level_param_context->named_values.begin(),
				top_level_param_context->named_values.end());
		func_utils::cbrackets = &cbrackets;

		BasicBlock* entry = BasicBlock::Create(llvm_module->getContext());
		oper_function->getBasicBlockList().push_back(entry);
		// clang-format off
		utils::generate_on_basic_block(entry, [&]() {
			std::function<void(const std::vector<ContextBracket>&)> create_allocas_for_values_over_runs =
					[&](const std::vector<ContextBracket>& brackets) {
						for(const func_utils::ContextBracket& bracket : brackets) {
							for(auto& [name, value] : bracket.values_over_all_runs->named_values) {
								if(value.value != nullptr)
									continue;
								value.value = ir_builder->CreateAlloca(utils::get_integer_type(FLX_VALUE_WIDTH),
										nullptr,
										utils::mkuniq(name + ".runs"));
								// allcoa mit dem Wert des Parameters aus dem ersten Vorkommnis initialisieren
								Value* param_value = ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), oper_function->getArg(value.index));
								ir_builder->CreateStore(param_value, value.value);
							}
							for(const ContextRun& run : bracket.runs)
								create_allocas_for_values_over_runs(run.brackets);
						}
					};
			create_allocas_for_values_over_runs(cbrackets);

			if(is_static) {
				BasicBlock* check_bb = BasicBlock::Create(llvm_module->getContext(), "check.present"_uniq);
				BasicBlock* does_not_exist_bb = BasicBlock::Create(llvm_module->getContext(), "not.present"_uniq);
				BasicBlock* does_exist = BasicBlock::Create(llvm_module->getContext(), "present"_uniq);

				Value* found_element;
				Value* stack_key;
				Value* actual_stack_key_len_v;

				static const int max_param_len = std::ceil(std::log10(std::pow(2, FLX_VALUE_WIDTH))) + 1;
				const int max_stack_key_len = oper_params.size() * max_param_len + names.str().size() + 1;

				// Alle Parameter-Werte so miteinander verknüpfen, dass ein eindeutiger Integer entsteht
				ir_builder->CreateBr(check_bb);
				oper_function->getBasicBlockList().push_back(check_bb);
				utils::generate_on_basic_block(check_bb, [&]() {
					// Hier kann der String auf den Stack angelegt werden, da search seine Adresse nicht speichern wird
					stack_key = ir_builder->CreateAlloca(ArrayType::get(utils::get_integer_type(CHARACTER_WIDTH), max_stack_key_len), nullptr, "stack.key"_uniq);
					utils::insert_into(stack_key, utils::get_integer_constant('\0', CHARACTER_WIDTH), "", 0,0);

					actual_stack_key_len_v = ir_builder->CreateAlloca(utils::get_integer_type());
					ir_builder->CreateStore(utils::get_integer_constant(names.str().size()), actual_stack_key_len_v);

					std::function<void(const std::vector<ContextBracket>&)> create_static_oper_id =
						[&](const std::vector<ContextBracket> cbrackets) {
							for(const ContextBracket& cbracket : cbrackets) {
								for(const ContextRun& run : cbracket.runs) {
									for(const auto& [_, value] : run.variables->named_values) {
										Value* param_value = ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), value.value);
										param_value = ir_builder->CreateZExt(param_value, utils::get_integer_type(STATIC_OPER_ID_WIDTH));

										Value* temp = ir_builder->CreateAlloca(ArrayType::get(utils::get_integer_type(CHARACTER_WIDTH), max_param_len));
										Value* param_str_len = ir_builder->CreateCall(predef_functions::sprintf(), {temp, utils::construct_string("%ld|"), param_value});
										Value* new_key_len = ir_builder->CreateLoad(utils::get_integer_type(), actual_stack_key_len_v);
										new_key_len = ir_builder->CreateAdd(new_key_len, param_str_len);
										ir_builder->CreateStore(new_key_len, actual_stack_key_len_v);

										ir_builder->CreateCall(predef_functions::strcat(), {stack_key, temp});
									}
									create_static_oper_id(run.brackets);
								}
							}
						};
					create_static_oper_id(cbrackets);
					ir_builder->CreateCall(predef_functions::strcat(), {stack_key, utils::construct_string(names.str())});

#ifdef RUNTIME_DEBUG_OUTPUT
					ir_builder->CreateCall(predef_functions::printf(), {utils::construct_string("Static oper id: \"%s\"\n"), stack_key});
#endif
					found_element = ir_builder->CreateCall(predef_functions::search(), stack_key);
					Value* found_something = ir_builder->CreateICmpNE(found_element, ConstantPointerNull::get(utils::get_ptr_type(predef_structures::hsearch_entry())));

					ir_builder->CreateCondBr(found_something, does_exist, does_not_exist_bb);
				});

				oper_function->getBasicBlockList().push_back(does_not_exist_bb);
				utils::generate_on_basic_block(does_not_exist_bb, [&](){
					// Implementierung generieren und am Schluss das Resultat dieser in die Map einfügen.
					// Dieser Block wird nur ausgeführt, wenn oben kein Eintrag gefunden wurde, also muss hier immer am Ende
					// ein neuer Eintrag erstellt werden.
					Value* oper_result;
					if(expr(row_)[CH::A + 4](passes_)[CH::A](branch_)[CH::A + 1](opnd_) == CH::nil)
						oper_result = flx_value::unique_synt();
					else
						oper_result = gen_expr_code_internal(expr(row_)[CH::A + 4](passes_)[CH::A](branch_)[CH::A + 1](opnd_));

					// Hier müssen key und data auf den Heap angelegt werden, da enter deren Adressen speichern wird
					Value* oper_result_ptr = ir_builder->CreateCall(predef_functions::malloc(), utils::get_integer_constant(std::ceil((float)(FLX_VALUE_WIDTH) / 8)));
					ir_builder->CreateStore(oper_result, oper_result_ptr);

					actual_stack_key_len_v = ir_builder->CreateLoad(utils::get_integer_type(), actual_stack_key_len_v);
					actual_stack_key_len_v = ir_builder->CreateAdd(actual_stack_key_len_v, utils::get_integer_constant(1)); // null-byte
					Value* heap_key = ir_builder->CreateCall(predef_functions::malloc(), actual_stack_key_len_v, "heap.key"_uniq);
					// https://llvm.org/docs/LangRef.html#llvm-memcpy-intrinsic
					std::vector<llvm::Type*> intrinsic_specialisation {
						PointerType::get(llvm_module->getContext(), DEFAULT_ADDRESS_SPACE),
						PointerType::get(llvm_module->getContext(), DEFAULT_ADDRESS_SPACE),
						utils::get_integer_type(32)
					};
					// Daten aus dem stack in den heap kopieren
					ir_builder->CreateCall(Intrinsic::getDeclaration(llvm_module, Intrinsic::memcpy, intrinsic_specialisation), {heap_key, stack_key, actual_stack_key_len_v});

					// Key und date in die map einfügen
					ir_builder->CreateCall(predef_functions::enter(), {heap_key, oper_result_ptr});

					ir_builder->CreateRet(oper_result);
				});

				oper_function->getBasicBlockList().push_back(does_exist);
				utils::generate_on_basic_block(does_exist, [&](){
					// Dieser Block wird nur ausgeführt, wenn oben ein Eintrag gefunden wurde
					Value* map_result = utils::extract_from_i(found_element, "", 0, 1);
					map_result = ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), map_result);

					ir_builder->CreateRet(map_result);
				});
			} else {
				if(expr(row_)[CH::A + 4](passes_)[CH::A](branch_)[CH::A + 1](opnd_) == CH::nil)
					ir_builder->CreateRet(flx_value::unique_synt());
				else {
					Value* res = gen_expr_code_internal(expr(row_)[CH::A + 4](passes_)[CH::A](branch_)[CH::A + 1](opnd_));
					ir_builder->CreateRet(res);
				}
			}
		}, true);
		// clang-format on

		func_utils::cbrackets = nullptr;
		cbrackets = cbracket_copy;
		verifyFunction(*oper_function);
	}

	out_cbrackets = cbrackets;

	return oper_function;
}

Value* get_ptr_to_current_bracket(const std::vector<std::pair<CH_id_t, Value*>>& indices_through_repetitions,
		const std::vector<Value*>& indices_through_brackets,
		std::vector<std::pair<BracketType, CH_id_t>> bracket_stack) {
	assert(!bracket_stack.empty());

	Function* enclosing_function = ir_builder->GetInsertBlock()->getParent();
	int next_index_through_repetitions = 0;
	int next_index_through_brackets = 0;
	CH_id_t id_to_search_for = bracket_stack.front().second;
	const auto& it = std::find_if(cbrackets->begin(),
			cbrackets->end(),
			[id_to_search_for](const ContextBracket& bracket) { return bracket.id == id_to_search_for; });
	assert(it != cbrackets->end());
	Value* ptr_to_bracket = enclosing_function->getArg(it->index_in_arg_list);

	for(int i = 1; i < bracket_stack.size(); ++i) {	 // Für alle geschachtelten Klammern
		std::vector<Value*> gep_args = {utils::get_integer_constant(0), utils::get_integer_constant(3)};
		Value* adr_of_ptr_to_runs = ir_builder->CreateGEP(predef_structures::bracket(), ptr_to_bracket, gep_args);
		Value* ptr_to_runs =
				ir_builder->CreateLoad(utils::get_ptr_type(predef_structures::bracket(), 3), adr_of_ptr_to_runs);
		if(bracket_stack.at(i - 1).first == repetition)
			gep_args = {indices_through_repetitions.at(next_index_through_repetitions++).second};
		else
			gep_args = {utils::get_integer_constant(0)};
		Value* ptr_correct_run =
				ir_builder->CreateGEP(utils::get_ptr_type(predef_structures::bracket(), 2), ptr_to_runs, gep_args);
		Value* correct_run =
				ir_builder->CreateLoad(utils::get_ptr_type(predef_structures::bracket(), 2), ptr_correct_run);
		// Hier Pre-Inkrement, da der Erste Wert übersprungen werden muss, da dieser sich auf die äußerste Klammer
		// bezieht und diese direkt aus den Argumenten der Funktion genommen wird und eben nicht durch diese Schleife
		// ermittelt wird
		gep_args = {indices_through_brackets.at(++next_index_through_brackets)};
		Value* ptr_to_correct_bracket =
				ir_builder->CreateGEP(utils::get_ptr_type(predef_structures::bracket()), correct_run, gep_args);
		ptr_to_bracket =
				ir_builder->CreateLoad(utils::get_ptr_type(predef_structures::bracket()), ptr_to_correct_bracket);
	}
	// indices_through_repetitions sollte genau so viele Elemente enthalten, wie es Wiederholungs-Klammern gab
	// next_index_index wird nur für Wiederholungs-Klammern erhöht.
	assert(next_index_through_repetitions == indices_through_repetitions.size());

	return ptr_to_bracket;
}
}

namespace opt {
Value* bracket_gen(Expr expr,
		std::vector<std::pair<func_utils::ContextBracket*, std::vector<std::pair<Value*, int>>>> bracket_from_all_runs,
		std::vector<std::pair<func_utils::BracketType, CH_id_t>> bracket_stack) {
	// Wird ein Operator mehrmals aufgerufen, muss diese map davor gelöscht werden, da alle klammern dann wieder bei 0
	// anfangen müssen
	static std::map<CH_id_t, int> run_to_choose;
	static int level = -1;
	++level;
	CH_id_t current_id = expr(oper_)(orig_).id;

	std::vector<Expr> options;
	// Für alle Optionen einen Block erstellen und den zugehörigen Ausdruck speichern
	for(Item item : expr(row_)) {
		if(Expr opnd = item(opnd_); opnd != CH::nil) {
			options.push_back(opnd);
		}
	}

	int run_index = run_to_choose[current_id]++;
	switch(bracket_stack.back().first) {
		case func_utils::alternative: {
			int selected_option = bracket_from_all_runs.at(run_index).first->runs.at(0).selected_option;
			current_context = bracket_from_all_runs.at(run_index).first->runs.at(0).variables;
			gen_expr_code_internal(options.at(selected_option));
			if(level-- == 0)
				run_to_choose.clear();
			return flx_value::get(flx_value::nat, selected_option + 1);
		}
		case func_utils::option: {
			if(bracket_from_all_runs.at(0).first->runs.empty())
				return flx_value::get(flx_value::nil);

			int selected_option = bracket_from_all_runs.at(run_index).first->runs.at(0).selected_option;
			current_context = bracket_from_all_runs.at(run_index).first->runs.at(0).variables;
			gen_expr_code_internal(options.at(selected_option));
			if(level-- == 0)
				run_to_choose.clear();
			return flx_value::get(flx_value::nat, selected_option + 1);
		}
		case func_utils::repetition: {
			for(const func_utils::ContextRun& run : bracket_from_all_runs.at(run_index).first->runs) {
				current_context = run.variables;
				gen_expr_code_internal(options.at(run.selected_option));
			}
			if(level-- == 0)
				run_to_choose.clear();
			return flx_value::get(flx_value::nat, bracket_from_all_runs.at(run_index).first->runs.size());
		}
		default: assert_not_reached();
	}
}
}

int gen_expr_code(Expr expr,
		int flags,
		std::function<int(Module&)> pre_code_gen,
		std::function<int(Module&)> post_code_gen) {
	settings_flags = flags;
	// llvm_context, llvm_module und ir_builder dürfen zwischen einzelnen Aufrufen von gen_expr_code "ins Nirvana"
	// zeigen, da die Verwendung aller anderen Funktionen undefiniert ist, wenn sie nicht direkt (oder indirekt) von
	// gen_expr_code aufgerufen werden.
	std::unique_ptr<LLVMContext> llvm_context_temp = std::make_unique<LLVMContext>();
	std::unique_ptr<Module> llvm_module_temp = std::make_unique<Module>("MOSTflexiPL", *llvm_context_temp);
	std::unique_ptr<IRBuilder<>> ir_builder_temp = std::make_unique<IRBuilder<>>(*llvm_context_temp);
	llvm_module = llvm_module_temp.get();
	ir_builder = ir_builder_temp.get();
	current_context = ValueContext::mk_ptr();

	if(int return_value = pre_code_gen(*llvm_module); return_value != 0)
		return return_value;

	BasicBlock* main_entry = BasicBlock::Create(llvm_module->getContext());
	predef_functions::main()->getBasicBlockList().push_back(main_entry);
	utils::generate_on_basic_block(main_entry, [&expr]() {
		ir_builder->CreateCall(predef_functions::init_map());
		Value* res = gen_expr_code_internal(expr);
		ir_builder->CreateCall(predef_functions::destory_map());
		ir_builder->CreateRet(flx_value::truncate_to_raw_value(res));
	});

	verifyFunction(*predef_functions::main(), &errs());

	return post_code_gen(*llvm_module);
}

Value* gen_expr_code_internal(Expr expr) {
	ValueContext::Ptr backup = current_context;
	Value* result = expr(oper_)(code_gen_)(expr);
	current_context = backup;
	return result;
}

Value* sequ_gen(Expr expr) {
	Expr lhs = expr(row_)[CH::A + 1](opnd_);
	Expr rhs = expr(row_)[CH::Z](opnd_);
	gen_expr_code_internal(lhs);
	return gen_expr_code_internal(rhs);
}

Value* print_gen(Expr expr) {
	Item printee = expr(row_)[CH::A + 1];
	Value* printee_v = gen_expr_code_internal(printee(opnd_));

	Value* width_v;
	if(Expr width = expr(row_)[CH::Z](passes_)[CH::A](branch_)[CH::Z](opnd_))
		width_v = flx_value::truncate_to_raw_value(gen_expr_code_internal(width));
	else
		width_v = utils::get_integer_constant(0);

	Value* is_nat = flx_value::is(printee_v, flx_value::nat);
	// Ist der erste Operand nicht natürlich, wird ein leerer String ausgegeben
	Value* string = ir_builder->CreateSelect(is_nat,
			utils::construct_string("%*d\n", "prinf_format"),
#ifdef RUNTIME_DEBUG_OUTPUT
			utils::construct_string("no nat value\n")
#else
			utils::construct_string("\n", "empty_printf")
#endif
	);

	std::vector<Value*> args;
	args.push_back(string);
	args.push_back(width_v);
	args.push_back(flx_value::truncate_to_raw_value(printee_v));

	ir_builder->CreateCall(predef_functions::printf(), args);
	return printee_v;
}

Value* cdecl_gen(Expr expr) {
	Oper oper = expr(expt_)(opers_)[CH::A];
	Value* result;
	if(Expr initializer = oper(init_))
		result = gen_expr_code_internal(initializer);
	else
		result = flx_value::unique_synt();

	std::stringstream name;
	name << expr(row_)[CH::A](passes_)[CH::A](branch_)[CH::A](word_);
	Value* variable;
	// Falls eine Variable oder Konstante auf globaler MOSTflexiPL Ebene deklariert wurde,
	// muss sie auch in der IR global sein, damit sie in allen Operator-Funktionen verwendet werden kann
	if(ir_builder->GetInsertBlock()->getParent() == predef_functions::main()) {
		variable = new GlobalVariable(*llvm_module,
				utils::get_integer_type(FLX_VALUE_WIDTH),
				false,
				GlobalValue::PrivateLinkage,
				utils::get_integer_constant(0, FLX_VALUE_WIDTH),
				utils::mkuniq(name.str() + ".ptr"));
	} else
		variable = ir_builder->CreateAlloca(utils::get_integer_type(FLX_VALUE_WIDTH),
				nullptr,
				utils::mkuniq(name.str() + ".ptr"));
	Value* store = ir_builder->CreateStore(result, variable);
	current_context->named_values.try_emplace(name.str(), variable);

	return result;
}

Value* logic_gen(Expr expr) {
	Item oper_item = expr(row_)[CH::A + 1];
	Expr lhs = expr(row_)[CH::A](opnd_);
	char oper = oper_item(word_)[CH::A];
	Expr rhs = expr(row_)[CH::Z](opnd_);

	Value* lhs_v = gen_expr_code_internal(lhs);
	Value* rhs_v = gen_expr_code_internal(rhs);

	Value* lhs_v_tr = flx_value::truncate_to_raw_value(lhs_v, BOOL_WIDTH);
	Value* rhs_v_tr = flx_value::truncate_to_raw_value(rhs_v, BOOL_WIDTH);

	Value* result;
	switch(oper) {
		case '&': result = ir_builder->CreateAnd(lhs_v_tr, rhs_v_tr); break;
		case '|': result = ir_builder->CreateOr(lhs_v_tr, rhs_v_tr); break;
		default: assert_not_reached();
	}

	result = ir_builder->CreateSelect(result, predef_globals::trueval(), predef_globals::nil());
	return ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), result);
}

Value* neg_gen(Expr expr) {
	Value* rhs_v = gen_expr_code_internal(expr(row_)[CH::Z](opnd_));

	return flx_value::exec_if([&]() { return flx_value::is(rhs_v, flx_value::nil); },
			[]() {
				return ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), predef_globals::trueval());
			},
			[]() { return flx_value::get(flx_value::nil); });
}

Value* branch_gen(Expr expr) {
	Expr condition = expr(row_)[CH::A + 1](opnd_);
	Function* enclosing_function = ir_builder->GetInsertBlock()->getParent();

	BasicBlock* if_bb = BasicBlock::Create(llvm_module->getContext(), "if"_uniq);
	BasicBlock* then_bb = BasicBlock::Create(llvm_module->getContext(), "if.then"_uniq);
	std::vector<std::pair<BasicBlock*, BasicBlock*>> elseif_bbs;  // test und then Block zusammen
	BasicBlock* else_bb = BasicBlock::Create(llvm_module->getContext(), "if.else"_uniq);
	BasicBlock* merge_bb = BasicBlock::Create(llvm_module->getContext(), "if.merge"_uniq);

	// Der Branch und der if_bb Block sind nur hier, um schöneren IR Code zu erzeugen!
	// Für besser optimierten Code sollte man das wohl entfernen
	ir_builder->CreateBr(if_bb);

	// Alle elseif teile generieren, aber noch nicht in die umschließende Function einfügen
	for(Pass pass : expr(row_)[CH::A + 5](passes_))
		elseif_bbs.emplace_back(BasicBlock::Create(llvm_module->getContext(), "elseif"_uniq),
				BasicBlock::Create(llvm_module->getContext(), "elseif.then"_uniq));

	enclosing_function->getBasicBlockList().push_back(if_bb);
	utils::generate_on_basic_block(if_bb, [&]() {
		Value* is_cond_nil = flx_value::is(gen_expr_code_internal(condition), flx_value::nil);
		ir_builder->CreateCondBr(is_cond_nil, elseif_bbs.empty() ? else_bb : elseif_bbs.at(0).first, then_bb);
	});

	Value* then_result;
	Value* else_result;
	// Es ist nur ein elseif_result nötig, da maximal einer der elseif-Blöcke ausgeführt wird und damit auch maximal ein
	// Ergebnis produziert wird
	Value* elseif_result;

	enclosing_function->getBasicBlockList().push_back(then_bb);
	utils::generate_on_basic_block(then_bb, [&]() {
		then_result = gen_expr_code_internal(expr(row_)[CH::A + 4](opnd_));
		// gen_expr_code_internal könnte neue Blöcke erzeugt haben
		then_bb = ir_builder->GetInsertBlock();
		ir_builder->CreateBr(merge_bb);
	});

	for(int i = 0; i < *expr(row_)[CH::A + 5](passes_); ++i) {
		Pass pass = expr(row_)[CH::A + 5](passes_)[CH::A + i];
		BasicBlock* current_elseif_test_bb = elseif_bbs.at(i).first;
		BasicBlock* current_elseif_then_bb = elseif_bbs.at(i).second;
		BasicBlock* next_elseif_test_or_else_bb = i + 1 < elseif_bbs.size() ? elseif_bbs.at(i + 1).first : else_bb;

		enclosing_function->getBasicBlockList().push_back(current_elseif_test_bb);
		utils::generate_on_basic_block(current_elseif_test_bb, [&]() {
			Value* condition_v =
					flx_value::truncate_to_raw_value(gen_expr_code_internal(pass(branch_)[CH::A + 1](opnd_)),
							BOOL_WIDTH);
			ir_builder->CreateCondBr(condition_v, current_elseif_then_bb, next_elseif_test_or_else_bb);
		});
		enclosing_function->getBasicBlockList().push_back(current_elseif_then_bb);
		utils::generate_on_basic_block(current_elseif_then_bb, [&]() {
			elseif_result = gen_expr_code_internal(pass(branch_)[CH::A + 3](opnd_));
			// gen_expr_code_internal könnte neue Blöcke erzeugt haben
			elseif_bbs.at(i).second = ir_builder->GetInsertBlock();
			ir_builder->CreateBr(merge_bb);
		});
	}

	enclosing_function->getBasicBlockList().push_back(else_bb);
	utils::generate_on_basic_block(else_bb, [&]() {
		if(Expr else_expr = expr(row_)[CH::A + 6](passes_)[CH::A](branch_)[CH::A + 1](opnd_))
			else_result = gen_expr_code_internal(else_expr);
		else
			else_result = flx_value::get(flx_value::nil);
		else_bb = ir_builder->GetInsertBlock();
		ir_builder->CreateBr(merge_bb);
	});

	PHINode* phi;
	enclosing_function->getBasicBlockList().push_back(merge_bb);
	utils::generate_on_basic_block(merge_bb, [&]() {
		phi = ir_builder->CreatePHI(utils::get_integer_type(FLX_VALUE_WIDTH), 2 + elseif_bbs.size());
		phi->addIncoming(then_result, then_bb);
		phi->addIncoming(else_result, else_bb);
		for(const auto& [test_bb, then_bb] : elseif_bbs)
			phi->addIncoming(elseif_result, then_bb);
	});

	return phi;
}

Value* loop_gen(Expr expr) {
	BasicBlock* loop_body = BasicBlock::Create(llvm_module->getContext(), "loop.body"_uniq);
	BasicBlock* loop_end = BasicBlock::Create(llvm_module->getContext(), "loop.end"_uniq);
	std::vector<BasicBlock*> loop_items;
	for(int _ = 0; _ < *expr(row_)[CH::A](passes_)(branch_()); ++_)
		loop_items.push_back(BasicBlock::Create(llvm_module->getContext()));

	Value* i_ptr = ir_builder->CreateAlloca(utils::get_integer_type(), nullptr, "loop.iter.count.ptr"_uniq);
	ir_builder->CreateStore(utils::get_integer_constant(0), i_ptr);

	ir_builder->CreateBr(loop_body);
	ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(loop_body);
	utils::generate_on_basic_block(loop_body, [&]() {
		Value* i = ir_builder->CreateLoad(utils::get_integer_type(), i_ptr, "loop.iter.count"_uniq);
		Value* i_inc = ir_builder->CreateAdd(i, utils::get_integer_constant(1));
		ir_builder->CreateStore(i_inc, i_ptr);
		ir_builder->CreateBr(loop_items.at(0));
		for(int i = 0; i < *expr(row_)[CH::A](passes_)(branch_()); ++i) {
			Row row = expr(row_)[CH::A](passes_)(branch_())[CH::A + i];
			BasicBlock* end_or_next = loop_items.size() - 1 > i ? loop_items.at(i + 1) : loop_end;
			ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(loop_items.at(i));
			utils::generate_on_basic_block(loop_items.at(i), [&]() {
				Value* result = gen_expr_code_internal(row[CH::Z](opnd_));
				Value* is_cond_nil = flx_value::is(result, flx_value::nil);
				switch(row[CH::A](passes_)[CH::A](branch_)[CH::A](word_)[CH::A]) {
					case 'w':
						loop_items.at(i)->setName("loop.while"_uniq);
						ir_builder->CreateCondBr(is_cond_nil, loop_end, end_or_next);
						break;
					case 'u':
						loop_items.at(i)->setName("loop.until"_uniq);
						ir_builder->CreateCondBr(is_cond_nil, end_or_next, loop_end);
						break;
					case 'd':
						// Ein do-Teil wird ausgeführt, ohne, dass sein Ergebnis eine Auswirkung auf den Ablauf der
						// Schleife hat
						loop_items.at(i)->setName("loop.do"_uniq);
						break;
					default: assert_not_reached();
				}
			});
		}
		ir_builder->CreateBr(loop_body);
	});

	ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(loop_end);
	ir_builder->SetInsertPoint(loop_end);

	Value* i = ir_builder->CreateLoad(utils::get_integer_type(), i_ptr, "loop.iter.count"_uniq);
	return flx_value::promote_to_flx_value(i, flx_value::nat);
}

Value* read_gen(Expr) {
	Value* destination = ir_builder->CreateAlloca(utils::get_integer_type(), nullptr, "scanf.result"_uniq);
	std::vector<Value*> args;
	args.push_back(utils::construct_string("%d", "scanf_format"));
	args.push_back(destination);
	Value* scanf_res = ir_builder->CreateCall(predef_functions::scanf(), args);
	Value* loaded_destination = ir_builder->CreateLoad(utils::get_integer_type(), destination);
	Value* flx_destination = flx_value::promote_to_flx_value(loaded_destination, flx_value::nat);
	// Konnte scanf keinen Text der mit %d übereinstimmt lesen, wird 0 zurückgegeben
	Value* successful_scanf = ir_builder->CreateICmpNE(scanf_res, utils::get_integer_constant(0));
	return ir_builder->CreateSelect(successful_scanf, flx_destination, flx_value::get(flx_value::nil));
}

Value* cmp_gen(Expr expr) {
	BasicBlock* early_exit_bb = BasicBlock::Create(llvm_module->getContext(), "cmp.early.exit"_uniq);
	BasicBlock* merge_bb = BasicBlock::Create(llvm_module->getContext(), "cmp.merge"_uniq);

	Row items;
	traverseA(expr(row_), [&items](Item item) {
		items += item;
		return false;
	});

	if(*items % 2 == 0)
		items = items(CH::A + 1 | CH::Z);

	std::vector<std::pair<BasicBlock*, BasicBlock*>> blocks;
	for(int i = 2; i <= *items; i += 2)
		blocks.emplace_back(BasicBlock::Create(llvm_module->getContext(), "cmp"_uniq),
				BasicBlock::Create(llvm_module->getContext(), "cmp.if"_uniq));

	Value* lhs_v;
	Value* trueval;
	ir_builder->CreateBr(blocks.at(0).first);
	for(int i = 2; i <= *items; i += 2) {
		int block_index = (i / 2) - 1;
		const auto& [block, _] = blocks.at(block_index);
		ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(block);
		utils::generate_on_basic_block(block, [&]() {
			if(i == 2)
				// Da der erste Block immer ausgeführt wird, muss in diesem schon geladen werden, da dies im merge block
				// nicht gemacht werden kann, da vor einer PHI-Instruktion nichts stehen darf
				trueval = ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), predef_globals::trueval());
			lhs_v = gen_expr_code_internal(items[CH::A](opnd_));
			Value* rhs_v = gen_expr_code_internal(items[i * CH::A + 1](opnd_));
			blocks.at(block_index).first = ir_builder->GetInsertBlock();

			Value* lhs_tr = flx_value::truncate_to_raw_value(lhs_v);
			Value* rhs_tr = flx_value::truncate_to_raw_value(rhs_v);

			Value* diff = ir_builder->CreateSub(lhs_tr, rhs_tr, "cmp.diff"_uniq);
			Value* bothnat = ir_builder->CreateAnd(flx_value::is(lhs_v, flx_value::nat),
					flx_value::is(rhs_v, flx_value::nat),
					"cmp.bothnat"_uniq);
			Value* equal = ir_builder->CreateOr(ir_builder->CreateICmpEQ(lhs_v, rhs_v),
					ir_builder->CreateAnd(bothnat, ir_builder->CreateICmpEQ(diff, utils::get_integer_constant(0))),
					"cmp.equal"_uniq);
			Value* result = equal;

			CH::str oper = items[CH::A * i](word_);
			switch(oper[CH::A]) {
				case '<':
					result = ir_builder->CreateAnd(bothnat,
							ir_builder->CreateICmpSLT(diff, utils::get_integer_constant(0)));
					break;
				case '>':
					result = ir_builder->CreateAnd(bothnat,
							ir_builder->CreateICmpSGT(diff, utils::get_integer_constant(0)));
					break;
			}
			switch(oper[CH::A + 1]) {
				case '=': result = ir_builder->CreateOr(result, equal); break;
				case '/':
					// LLVM besitzt keine negation. XOR mit true erzielt allerdings dasselbe
					result = ir_builder->CreateXor(result, utils::get_integer_constant(1, BOOL_WIDTH));
					break;
			}

			ir_builder->CreateBr(blocks.at(block_index).second);
			ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(blocks.at(block_index).second);
			utils::generate_on_basic_block(blocks.at(block_index).second, [&]() {
				BasicBlock* next_or_merge =
						blocks.size() == block_index + 1 ? merge_bb : blocks.at(block_index + 1).first;
				ir_builder->CreateCondBr(result, next_or_merge, early_exit_bb);
			});
			lhs_v = rhs_v;
		});
	}

	ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(early_exit_bb);
	utils::generate_on_basic_block(early_exit_bb, [&]() { ir_builder->CreateBr(merge_bb); });

	PHINode* phi;
	ir_builder->GetInsertBlock()->getParent()->getBasicBlockList().push_back(merge_bb);
	utils::generate_on_basic_block(merge_bb, [&]() {
		phi = ir_builder->CreatePHI(utils::get_integer_type(FLX_VALUE_WIDTH), blocks.size());
		phi->addIncoming(flx_value::get(flx_value::nil), early_exit_bb);
		phi->addIncoming(trueval, blocks.back().second);
	});

	return phi;
}

Value* bin_gen(Expr expr) {
	Item oper_item = expr(row_)[CH::A + 1];
	Expr lhs = expr(row_)[CH::A](opnd_);
	char oper = oper_item(word_)[CH::A];
	Expr rhs = expr(row_)[CH::A + 2](opnd_);

	if(!oper_item(word_))
		oper = oper_item(passes_)[CH::A](branch_)[CH::A](word_)[CH::A];

	Value* lhs_v = gen_expr_code_internal(lhs);
	Value* rhs_v = gen_expr_code_internal(rhs);

	return flx_value::exec_if(
			[&]() {
				Value* is_lhs_nat_v = flx_value::is(lhs_v, flx_value::nat);
				Value* is_rhs_nat_v = flx_value::is(rhs_v, flx_value::nat);
				return ir_builder->CreateAnd(is_lhs_nat_v, is_rhs_nat_v);
			},
			[&]() {
				Value* lhs_v_tr = flx_value::truncate_to_raw_value(lhs_v);
				Value* rhs_v_tr = flx_value::truncate_to_raw_value(rhs_v);

				Value* result;
				switch(oper) {
					case '+': result = ir_builder->CreateAdd(lhs_v_tr, rhs_v_tr); break;
					case '-': result = ir_builder->CreateSub(lhs_v_tr, rhs_v_tr); break;
					case '*': result = ir_builder->CreateMul(lhs_v_tr, rhs_v_tr); break;
					case '/': {
						// Wird durch 0 geteilt, muss nil zurückgegeben werden
						Value* is_rhs_zero = ir_builder->CreateICmpEQ(rhs_v_tr, utils::get_integer_constant(0));
						result = ir_builder->CreateSelect(is_rhs_zero,
								flx_value::get(flx_value::nil),
								ir_builder->CreateSDiv(lhs_v_tr, rhs_v_tr));
					} break;
					case '^': {
						if(settings_flags & O1 && isa<ConstantInt>(lhs_v_tr) && isa<ConstantInt>(rhs_v_tr)) {
							int64_t lhs_i = cast<ConstantInt>(lhs_v_tr)->getSExtValue();
							int64_t rhs_i = cast<ConstantInt>(rhs_v_tr)->getSExtValue();
							result = flx_value::get(flx_value::nat, static_cast<int>(std::pow(lhs_i, rhs_i)));
						} else {
							std::vector<Value*> args;
							args.push_back(lhs_v);
							args.push_back(rhs_v);
							result = ir_builder->CreateCall(predef_functions::pow(), args);
						}
						break;
					}
					default: assert_not_reached();
				}
				return flx_value::promote_to_flx_value(result, flx_value::nat, false);
			},
			[]() { return flx_value::get(flx_value::nil); });
}

Value* chs_gen(Expr expr) {
	Value* rhs_v = gen_expr_code_internal(expr(row_)[CH::Z](opnd_));

	return flx_value::exec_if([&]() { return flx_value::is(rhs_v, flx_value::nat); },
			[&]() {
				Value* rhs_tr = flx_value::truncate_to_raw_value(rhs_v);
				return flx_value::promote_to_flx_value(ir_builder->CreateSub(utils::get_integer_constant(0), rhs_tr),
						flx_value::nat);
			},
			[]() { return flx_value::get(flx_value::nil); });
}

Value* fac_gen(Expr expr) {
	Value* rhs_v = gen_expr_code_internal(expr(row_)[CH::A](opnd_));

	return flx_value::exec_if([&]() { return flx_value::is(rhs_v, flx_value::nat); },
			[&]() -> Value* {
				Value* rhs_v_tr = flx_value::truncate_to_raw_value(rhs_v);

				if(settings_flags & O1 && isa<ConstantInt>(rhs_v_tr)) {
					int rhs_i = static_cast<int>(cast<ConstantInt>(rhs_v_tr)->getSExtValue());
					int z = 1;
					while(rhs_i > 1)
						z *= rhs_i--;
					return flx_value::get(flx_value::nat, z);
				}

				std::vector<Value*> args(1, rhs_v);
				return ir_builder->CreateCall(predef_functions::fac(), args);
			},
			[]() { return flx_value::get(flx_value::nil); });
}

Value* paren_gen(Expr expr) { return gen_expr_code_internal(expr(row_)[CH::A + 2](opnd_)); }

Value* intlit_gen(Expr expr) {
	int32_t int_val = 0;
	for(char c : expr(row_)[CH::A](word_))
		int_val = int_val * 10 + c - '0';

	return flx_value::get(flx_value::nat, int_val);
}

Value* boollit_gen(Expr expr) {
	CH::posA choice = expr(row_)[CH::A](passes_)[CH::A](choice_);
	return flx_value::get(flx_value::nat, choice == CH::A ? 1 : 0);
}

Value* const_gen(Expr expr) {
	if(expr(row_)[CH::A + 1](word_) == "nil")
		return flx_value::get(flx_value::nil);
	std::stringstream name;
	name << expr(row_)[CH::A](word_);
	ValueContext::ValueOrArgIndex* alloca = current_context->lookup_value(name.str());
	// Wenn in einer Impl-Klammer eine Konstante verwendet wird die Klammer aber nicht durchlaufen wird,
	// wird diese hier nicht gefunden.
	// Parameter eckiger Klammern mit nur Alternative bekommen keinen eigenen Kontext und
	// sind auch außerhalb der Klammer sichtbar, weswegen auf sie im Gegensatz zu allen anderen Parametern während der
	// Laufzeit tatsächlich zugegriffen werden kann, auch wenn die Klammer nicht durchlaufen wurde. Deswegen muss hier
	// für genau diesen Fall der wohldefinierte Wert nil zurückgegeben werden.
	if(alloca == nullptr) {
#ifdef COMPILE_TIME_DEBUG_OUTPUT
		std::cout << "Konstante " << name.str() << " nicht gefunden!" << std::endl;
#endif
		return flx_value::get(flx_value::nil);
	}
	return ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), alloca->value, utils::mkuniq(name.str()));
}

Value* odecl_gen(Expr expr) {
	bool is_static = expr(expt_)(opers_)[CH::A](orig_)(stat_);
	const std::function<std::stringstream(Expr)> gen_function_name = [&](Expr expr) -> std::stringstream {
		static int level = -1;
		++level;
		std::stringstream name;
		for(Pass pass : expr(row_)[CH::A](passes_)) {
			int index;
			switch(pass(choice_) - CH::A) {
				case 0:	 // Name
					if(level == 0)
						name << pass(branch_)[CH::A](word_);
					break;
				default: name << gen_function_name(pass(branch_)[CH::A + 2](opnd_)).str();	// u[0]
			}
		}
		--level;
		return name;
	};

	std::stringstream name;
	if(settings_flags & READABLE_FUNCTION_NAMES)
		name = gen_function_name(expr);
	name << (void*)expr(expt_)(opers_)[CH::A](orig_).id;
	func_utils::available_functions_to_instantiate.try_emplace(name.str(), is_static, expr);

	return flx_value::get(flx_value::nil);
}

ATTR1(oper_, Part, Oper)
Value* appl_gen(Expr expr) {
	int num_args = 0;
	std::vector<Value*> oper_brackets;
	std::vector<Value*> oper_params;
	std::vector<func_utils::AllocaBracket> abrackets;
	func_utils::AllocaBracket* current_abracket = nullptr;
	std::stringstream function_name;

	std::function<Value*(CH::seq<Item>, CH::seq<Part>)> generate_call_args = [&](CH::seq<Item> items,
																					 CH::seq<Part> parts) -> Value* {
		// Falls EXPAND_BRACKETS gesetzt ist, sollen keine bracket Objekte erstellt werden.
		// Diese werden nicht benötigt, da alle Implementierungsklammern zur Übersetzungszeit expandiert werden.
		const bool generate_brackets = !(settings_flags & EXPAND_BRACKETS);
		static int level = -1;
		++level;
		Value* bracket_ptr = nullptr;
		std::vector<Value*> bracket_ptrs;
		for(int i = 0; i < *items; ++i) {
			Item item = items[CH::A + i];
			Part part = parts[CH::A + i];

			if(CH::str word = item(word_)) {
				if(level == 0 && settings_flags & READABLE_FUNCTION_NAMES)
					function_name << word;
			}
			if(Par param = part(par_)(orig_)) {
				Value* res;
				// Hat ein Parameter einen Operanden, spielt seine default-Initialisierung keine Rolle mehr
				if(Expr opnd = item(opnd_))
					res = gen_expr_code_internal(opnd);
				else if(Expr default_init = param(init_))
					res = gen_expr_code_internal(default_init);
				else
					res = flx_value::get(flx_value::nil);
				AllocaInst* param_alloca = ir_builder->CreateAlloca(res->getType());
				ir_builder->CreateStore(res, param_alloca);
				if(level == 0)
					oper_params.push_back(param_alloca);
				else
					current_abracket->runs.back().variables.push_back(param_alloca);
				++num_args;
			}

			//  Wenn rep_ gesetzt ist, ist auch opt_ gesetzt. Also wird rep_ hier nicht überprüft
			if(part(opt_) || part(alts_)) {
				CH::seq<Pass> passes = item(passes_);
				func_utils::AllocaBracket* old_current_abracket = current_abracket;
				if(current_abracket != nullptr)
					current_abracket =
							&current_abracket->runs.back().brackets.emplace_back(part(oper_)(orig_).id, *part(alts_));
				else {
					current_abracket = &abrackets.emplace_back(part(oper_)(orig_).id, *part(alts_));
					++num_args;
				}
				if(part(rep_)) {
					if(generate_brackets) {
						bracket_ptr = ir_builder->CreateAlloca(predef_structures::bracket(), nullptr, "mult.ptr"_uniq);
						utils::insert_into(bracket_ptr, *passes, "mult.num.passes"_uniq, 0, 1);
					}
					Value* selected_option_ptr = nullptr;
					Value* runs_ptr = nullptr;
					for(int j = 0; j < *passes; ++j) {
						Pass pass = passes[CH::A + j];
						current_abracket->runs.emplace_back(pass(choice_) - CH::A);
						Value* brackets = generate_call_args(pass(branch_), part(alts_)[pass(choice_)]);
						if(runs_ptr == nullptr && generate_brackets) {
							// Speicher für *passes viele Zeiger auf die jeweiligen Durchläufe reservieren
							runs_ptr = ir_builder->CreateAlloca(utils::get_ptr_type(predef_structures::bracket(), 2),
									utils::get_integer_constant(*passes),
									"mult.brackets.ptr"_uniq);
						}
						if(brackets != nullptr && generate_brackets)
							utils::insert_into(runs_ptr, brackets, "", j);
						if(selected_option_ptr == nullptr && generate_brackets) {
							// Speicher für *passes viele Indizes für die jeweils gewählten Option reservieren
							selected_option_ptr = ir_builder->CreateAlloca(utils::get_integer_type(),
									utils::get_integer_constant(*passes),
									"mult.brackets.sel"_uniq);
						}
						if(generate_brackets)
							utils::insert_into(selected_option_ptr, pass(choice_) - CH::A, "mult.selopt"_uniq, j);
					}
					// *passes > 0, da andernfalls selected_option_ptr und runs_ptr null sind
					if(*passes > 0 && generate_brackets) {
						utils::insert_into(bracket_ptr, selected_option_ptr, "mult.selopt.ptr"_uniq, 0, 0);
						utils::insert_into(bracket_ptr, runs_ptr, "mult.brackets.ptr"_uniq, 0, 3);
					}
				} else if(part(opt_)) {
					current_abracket->runs.emplace_back(item(passes_)[CH::A](choice_) - CH::A);
					if(generate_brackets)
						bracket_ptr = ir_builder->CreateAlloca(predef_structures::bracket(), nullptr, "opt.ptr"_uniq);
					int selected_option = item(passes_)[CH::A](choice_) - CH::A;
					bool has_option = item(passes_)[CH::A](choice_) != CH::A * 0;
					Value* brackets =
							generate_call_args(item(passes_)[CH::A](branch_), part(alts_)[CH::A + selected_option]);
					Value* runs_ptr;
					if(brackets != nullptr && generate_brackets) {
						// Speicher für einen Zeiger auf den jeweiligen Durchlauf reservieren
						runs_ptr = ir_builder->CreateAlloca(utils::get_ptr_type(predef_structures::bracket(), 2),
								utils::get_integer_constant(1),
								"opt.brackets.ptr"_uniq);
					}
					if(generate_brackets) {
						// Speicher für einen Index der gewählten Option reservieren
						Value* selected_option_ptr = ir_builder->CreateAlloca(utils::get_integer_type(),
								utils::get_integer_constant(1),
								"opt.brackets.sel"_uniq);

						if(brackets != nullptr) {  // Gab es geschachtelte Klammern werden diese eingefügt
							utils::insert_into(runs_ptr, brackets, "opt.brackets"_uniq, 0);
							utils::insert_into(bracket_ptr, runs_ptr, "opt.brackets.ptr"_uniq, 0, 3);
						}
						utils::insert_into(bracket_ptr, has_option, "opt.has.opt"_uniq, 0, 2);
						utils::insert_into(selected_option_ptr, selected_option, "opt.selopt"_uniq, 0);
						utils::insert_into(bracket_ptr, selected_option_ptr, "opt.selopt.ptr"_uniq, 0, 0);
					}
				} else if(part(alts_)) {
					current_abracket->runs.emplace_back(item(passes_)[CH::A](choice_) - CH::A);
					if(generate_brackets)
						bracket_ptr = ir_builder->CreateAlloca(predef_structures::bracket(), nullptr, "alt.ptr"_uniq);
					int selected_alternative = item(passes_)[CH::A](choice_) - CH::A;
					Value* brackets = generate_call_args(item(passes_)[CH::A](branch_),
							part(alts_)[CH::A + selected_alternative]);
					Value* runs_ptr;
					if(brackets != nullptr && generate_brackets) {
						// Speicher für einen Zeiger auf den jeweiligen Durchlauf reservieren
						runs_ptr = ir_builder->CreateAlloca(utils::get_ptr_type(predef_structures::bracket(), 2),
								utils::get_integer_constant(1),
								"alt.brackets.ptr"_uniq);
					}
					if(generate_brackets) {
						// Speicher für einen Index der gewählten Option reservieren
						Value* selected_alternative_ptr = ir_builder->CreateAlloca(utils::get_integer_type(),
								utils::get_integer_constant(1),	 // 1 da es nur einen Durchlauf gibt
								"alt.selalt.ptr"_uniq);

						if(brackets != nullptr) {  // Gab es geschachtelte Klammern werden diese eingefügt
							utils::insert_into(runs_ptr, brackets, "", 0);
							utils::insert_into(bracket_ptr, runs_ptr, "", 0, 3);
						}
						utils::insert_into(selected_alternative_ptr, selected_alternative, "", 0);
						utils::insert_into(bracket_ptr, selected_alternative_ptr, "", 0, 0);
					}
				} else
					assert_not_reached();
				if(bracket_ptr == nullptr) {
					assert(settings_flags & EXPAND_BRACKETS);
					bracket_ptr = ConstantPointerNull::get(utils::get_ptr_type(predef_structures::bracket()));
				}
				if(level == 0) {
					oper_brackets.push_back(bracket_ptr);
					current_abracket->bracket_ptr = bracket_ptr;
				}
				current_abracket = old_current_abracket;
				bracket_ptrs.push_back(bracket_ptr);
			}
		}
		--level;

		if(bracket_ptrs.empty() || !generate_brackets)
			return nullptr;

		Value* brackets = ir_builder->CreateAlloca(utils::get_ptr_type(predef_structures::bracket()),
				utils::get_integer_constant(bracket_ptrs.size()),
				"run.brackets"_uniq);
		for(int i = 0; i < bracket_ptrs.size(); ++i)
			utils::insert_into(brackets, bracket_ptrs.at(i), "", i);
		return brackets;
	};
	generate_call_args(expr(row_), expr(oper_)(sig_));
	function_name << (void*)expr(oper_)(orig_).id;

	current_context = ValueContext::mk_ptr(current_context);
	std::vector<func_utils::ContextBracket> cbrackets;
	Function* instantiated_operator = func_utils::instantiate_function(function_name.str(), abrackets, cbrackets);

	std::vector<Value*> oper_args(num_args, nullptr);
	std::function<void(const std::vector<func_utils::ContextBracket>&)> insert_bracket_and__param_allocas =
			[&oper_args, &insert_bracket_and__param_allocas](const std::vector<func_utils::ContextBracket>& cbrackets) {
				for(const func_utils::ContextBracket& cbracket : cbrackets) {
					if(cbracket.index_in_arg_list != -1) {	// nur für top-level Klammern
						assert(oper_args.at(cbracket.index_in_arg_list) == nullptr);
						oper_args.at(cbracket.index_in_arg_list) = cbracket.bracket_ptr;
					} else
						assert(cbracket.bracket_ptr == nullptr);
					for(const func_utils::ContextRun& run : cbracket.runs) {
						insert_bracket_and__param_allocas(run.brackets);
						for(const auto& [name, value] : run.variables->named_values) {
							assert(value.index != -1 && value.outside_alloca != nullptr &&
									oper_args.at(value.index) == nullptr);
							oper_args.at(value.index) = value.outside_alloca;
						}
					}
				}
			};
	const auto& insert_top_level_param_allocas = [&](const std::vector<Value*>& params) {
		int param_index = 0;
		for(int i = 0; i < oper_args.size(); ++i) {
			if(oper_args.at(i) != nullptr)
				continue;
			// Steht an der aktuellen Stelle ein null-Zeiger, wird dort ein Parameter eingesetzt und der Index für den
			// nächste Parameter um eins erhöht
			oper_args.at(i) = params.at(param_index++);
		}
	};

	insert_bracket_and__param_allocas(cbrackets);
	// Nachdem alle geschachtelten Parameter und top-level Klammern eingefügt wurden, steht an genau den Stellen ein
	// null Zeiger, an die eine top-level Parameter muss. Ergo muss insert_top_level_param_allocas nach
	// insert_param_allocas aufgerufen werden
	insert_top_level_param_allocas(oper_params);

	return ir_builder->CreateCall(instantiated_operator, oper_args);
}

Value* bracket_gen(Expr expr) {
	assert(func_utils::cbrackets != nullptr);

	static std::vector<std::pair<func_utils::BracketType, CH_id_t>> bracket_stack;

	// clang-format off
	using func_utils::alternative;
	using func_utils::repetition;
	using func_utils::option;
	// clang-format on

	CH_id_t current_id = expr(oper_)(orig_).id;
	switch(expr(row_)[CH::A](word_)[CH::A]) {
		case '(': bracket_stack.emplace_back(alternative, current_id); break;
		case '{': bracket_stack.emplace_back(repetition, current_id); break;
		case '[': bracket_stack.emplace_back(option, current_id); break;
		default: assert_not_reached();
	}

	const auto& get_bb_name_with_suffix = [](const std::string& suffix) {
		std::string bracket_prefix;
		switch(bracket_stack.back().first) {
			case alternative: bracket_prefix = "alt."; break;
			case repetition: bracket_prefix = "mult."; break;
			case option: bracket_prefix = "opt."; break;
			default: assert_not_reached();
		}
		return bracket_prefix + suffix;
	};

	Function* enclosing_function = ir_builder->GetInsertBlock()->getParent();
	Value* return_value;

	struct Option {
		Option(Expr expression, BasicBlock* block) : expression(expression), block(block) {}
		Expr expression;
		BasicBlock* block;
	};

	std::vector<Option> options;
	// Für alle Optionen einen Block erstellen und den zugehörigen Ausdruck speichern
	for(Item item : expr(row_)) {
		if(item(opnd_)) {
			options.emplace_back(item(opnd_),
					BasicBlock::Create(llvm_module->getContext(), utils::mkuniq(get_bb_name_with_suffix("option"))));
		}
	}

	static std::vector<std::pair<CH_id_t, Value*>> iterations_through_repetitions;
	const auto& find_index_through_rep_by_id = [&](CH_id_t id_to_search_for) -> Value* {
		for(const auto& [id, value] : iterations_through_repetitions) {
			if(id == id_to_search_for)
				return value;
		}
		return nullptr;
	};

	// Hier werden alle dynamischen Vorkommnisse einer Klammer gespeichert. Ist eine Klammer in einer oder mehrer
	// Wiederholungen enthalten sind also meistens auch mehrere Elemente drin.
	// Zusätzlich zu den Klammer-Objekten selbst wird auch eine Liste gespeichert, die beschreibt wie man zu dieser
	// Klammer kommt. Also in welchem Durchlauf welcher umschließenden Klammer sie sich befindet. Dabei enthält Value*
	// den Index des aktuellen Durchlaufs und int den Index des Durchlaufs in dem die Klammer tatsächlich vorkam
	std::vector<std::pair<func_utils::ContextBracket*, std::vector<std::pair<Value*, int>>>> bracket_from_all_runs;
	std::function<void(std::vector<func_utils::ContextBracket>&)> setup_bracket_from_all_runs =
			[&](std::vector<func_utils::ContextBracket>& contexts) {
				static std::vector<std::pair<Value*, int>> temp;
				for(func_utils::ContextBracket& bracket : contexts) {
					bool already_present = false;
					func_utils::ContextBracket* last = nullptr;
					for(const auto& [brackert, _] : bracket_from_all_runs) {
						if(last != brackert)
							continue;
						already_present = true;
						break;
					}
					if(Value* index = find_index_through_rep_by_id(bracket.id); index != nullptr)
						temp.emplace_back(index, -1);
					if(bracket.id == current_id) {
						if(!already_present)
							bracket_from_all_runs.emplace_back(&bracket, temp);
					} else {
						for(int i = 0; i < bracket.runs.size(); ++i) {
							if(find_index_through_rep_by_id(bracket.id) != nullptr)
								temp.back().second = i;
							setup_bracket_from_all_runs(bracket.runs.at(i).brackets);
						}
					}
					if(find_index_through_rep_by_id(bracket.id) != nullptr)
						temp.pop_back();
				}
			};
	setup_bracket_from_all_runs(*func_utils::cbrackets);

	// current_run_index_v ist nur gesetzt, falls die aktuelle Klammer eine Wiederholung ist.
	// Dann zeigt dieser Parameter auf den Index des aktuellen Durchlaufs der Klammer
	const auto& setup_selects = [&](Value* current_run_index_v = nullptr) {
		if(bracket_from_all_runs.at(0).first->values_over_all_runs->named_values.empty())
			return;	 // Enthält die Klammer in keinem Durchlauf Variable, kann man sich das alles sparen
		// Ist reps in der ersten Iteration leer, sollte es in allen Iterationen leer sein, da eine Klammer nicht
		// plötzliche andere umschließende Klammern hat
		if(bracket_from_all_runs.at(0).second.empty() && current_run_index_v == nullptr)
			// Ist die aktuelle Klammer nicht in mindestens einer Wiederholungs-Klammer enthalten, muss kein extra Code
			// generiert werden, da die Parameter im voar-Kontext automatisch auf das erste Vorkommnis gesetzt sind.
			// Das gilt allerdings nicht, wenn die Klammer selbst eine Wiederholung ist (current_run_index_v != nullptr)
			return;
		for(const auto& [bracket, reps] : bracket_from_all_runs) {
			Value* acc = utils::get_integer_constant(1, BOOL_WIDTH);
			// Ist die aktuelle Klammer eine Wiederholung, enthält reps die aktuelle Klammer nicht!
			// Da diese Schleife über alle umschließenden Klammern iterieren soll, wird im Falle einer
			// Wiederholungs-Klammer einfach die Länge als Grenze genommen, während bei allen anderen Klammern 1
			// abgezogen werden muss (da in allen anderen Klammern die aktuelle Klammer ja auch noch in der Liste
			// enthalten ist)
			int upper_bound = current_run_index_v ? reps.size() : (int)reps.size() - 1;
			for(int i = 0; i < upper_bound; ++i) {
				const auto& [index_v, used_index] = reps.at(i);
				Value* equal = ir_builder->CreateICmpEQ(index_v, utils::get_integer_constant(used_index));
				acc = ir_builder->CreateAnd(acc, equal);
			}
			// Hier wird für alle Parameter aus allen Durchläufen der jeweils korrekte Wert ausgewählt
			for(const auto& [name, value] : bracket->values_over_all_runs->named_values) {
				Value* value_to_use = value.value;
				for(int i = 0; i < bracket->runs.size(); ++i) {
					const auto& named_values = bracket->runs.at(i).variables->named_values;
					Value* actual_run = current_run_index_v ? current_run_index_v : reps.back().first;
					Value* target_run = utils::get_integer_constant(current_run_index_v ? i : reps.back().second);
					Value* value_to_use_for_target_run;
					// Ist der Parameter in dem aktuellen Durchlauf nicht gesetzt (da eine andere Alternative gewählt
					// wurde, oder eine optionale Klammer gar nicht durchlaufen wurde), wird er auf nil gesetzt
					if(const auto& it = named_values.find(name); it == named_values.end())
						value_to_use_for_target_run = predef_globals::nil();
					else
						value_to_use_for_target_run = it->second.value;
					Value* is_target_run = ir_builder->CreateICmpEQ(actual_run, target_run);
					value_to_use = ir_builder->CreateSelect(ir_builder->CreateAnd(acc, is_target_run),
							value_to_use_for_target_run,
							value_to_use);
				}
				value_to_use = ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), value_to_use);
				ir_builder->CreateStore(value_to_use, value.value);
			}
		}
	};

	std::vector<int> bracket_indices;
	std::function<bool(std::vector<func_utils::ContextBracket>&)> setup_bracket_indices =
			[&](std::vector<func_utils::ContextBracket>& cbrackets) {
				bracket_indices.push_back(-1);
				for(int i = 0; i < cbrackets.size(); ++i) {
					if(cbrackets.at(i).id == current_id) {
						bracket_indices.back() = i;
						return true;
					} else {
						for(int j = 0; j < cbrackets.at(i).runs.size(); ++j) {
							if(cbrackets.at(i).runs.at(j).brackets.empty())
								continue;
							bracket_indices.back() = i;
							if(setup_bracket_indices(cbrackets.at(i).runs.at(j).brackets)) {
								return true;
							}
						}
					}
				}
				return false;
			};

	setup_bracket_indices(*func_utils::cbrackets);

	if(settings_flags & EXPAND_BRACKETS)
		return opt::bracket_gen(expr, bracket_from_all_runs, bracket_stack);

	bool all_runs_empty = true;
	for(const auto& [bracket, _] : bracket_from_all_runs) {
		if(bracket->runs.empty())
			continue;
		all_runs_empty = false;
		break;
	}
	// Hat die aktuelle Klammer in allen Vorkommnissen keine Durchläufe (all_runs_empty),
	// oder kam sie selbst gar nie vor (bracket_from_all_runs.empty()) muss kein Code generiert werden
	if(bracket_from_all_runs.empty() || all_runs_empty) {
#ifdef COMPILE_TIME_DEBUG_OUTPUT
		std::cout << "bracket_gen early exit" << std::endl;
#endif
		bracket_stack.pop_back();
		return flx_value::get(flx_value::nil);
	}

	std::vector<Value*> indices_through_brackets;
	for(int index : bracket_indices)
		indices_through_brackets.push_back(utils::get_integer_constant(index));

	Value* ptr_to_current_bracket = func_utils::get_ptr_to_current_bracket(iterations_through_repetitions,
			indices_through_brackets,
			bracket_stack);

	ValueContext::Ptr old_context = current_context;
	switch(bracket_stack.back().first) {
		case alternative: {
			setup_selects();
			BasicBlock* select_alternative_bb = BasicBlock::Create(llvm_module->getContext(), "alt.select.alt"_uniq);
			BasicBlock* merge_alternatives_bb = BasicBlock::Create(llvm_module->getContext(), "alt.merge.alt"_uniq);

			Value* chosen_alternative_per_run_v = utils::extract_from_i(ptr_to_current_bracket, "", 0, 0);
			Value* chosen_alternative_v = utils::extract_from_i(chosen_alternative_per_run_v, "alt.chosen.alt"_uniq, 0);

			ir_builder->CreateBr(select_alternative_bb);
			enclosing_function->getBasicBlockList().push_back(select_alternative_bb);
			utils::generate_on_basic_block(select_alternative_bb, [&]() {
				SwitchInst* switch_inst =
						ir_builder->CreateSwitch(chosen_alternative_v, merge_alternatives_bb, options.size());
				for(int i = 0; i < options.size(); ++i)
					switch_inst->addCase(utils::get_integer_constant(i), options.at(i).block);
			});

			// values_over_all_runs zeigt auf denselben Kontext, egal welches Element aus
			// bracket_from_all_runs ausgewählt wird
			current_context = bracket_from_all_runs.at(0).first->values_over_all_runs;
			for(Option& alternative : options) {
				enclosing_function->getBasicBlockList().push_back(alternative.block);
				utils::generate_on_basic_block(alternative.block, [&alternative, &merge_alternatives_bb]() {
					gen_expr_code_internal(alternative.expression);
					// gen_expr_code_internal könnte neue Blöcke erzeugt haben
					// wobei das Aktualisieren hier eigentlich gar nicht relevant ist, da die Blöcke hiernach
					// nicht mehr benötigt werden
					alternative.block = ir_builder->GetInsertBlock();
					ir_builder->CreateBr(merge_alternatives_bb);
				});
			}

			enclosing_function->getBasicBlockList().push_back(merge_alternatives_bb);
			utils::generate_on_basic_block(merge_alternatives_bb, [&]() {
				Value* chosen_alternative_plus_one_v =
						flx_value::promote_to_flx_value(ir_builder->CreateAdd(chosen_alternative_v,
																utils::get_integer_constant(1),
																"alt.chosen.alt.inc"_uniq),
								flx_value::nat);
				return_value = chosen_alternative_plus_one_v;
			});
			break;
		}
		case repetition: {
			BasicBlock* loop_begin_bb = BasicBlock::Create(llvm_module->getContext(), "mult.loop.begin"_uniq);
			BasicBlock* loop_body_bb = BasicBlock::Create(llvm_module->getContext(), "mult.loop.body"_uniq);
			BasicBlock* loop_opt_merge = BasicBlock::Create(llvm_module->getContext(), "mult.merge.opt"_uniq);
			BasicBlock* loop_end_bb = BasicBlock::Create(llvm_module->getContext(), "mult.loop.end"_uniq);
			bool needs_switch = options.size() > 1 || settings_flags & O0;

			Value* number_of_runs_v = utils::extract_from_i(ptr_to_current_bracket, "mult.runs"_uniq, 0, 1);

			ir_builder->CreateBr(loop_begin_bb);

			PHINode* run_index_v = nullptr;
			BasicBlock* current_bb = ir_builder->GetInsertBlock();
			enclosing_function->getBasicBlockList().push_back(loop_begin_bb);
			utils::generate_on_basic_block(loop_begin_bb, [&]() {
				run_index_v = ir_builder->CreatePHI(utils::get_integer_type(), 2, "run.index"_uniq);
				run_index_v->addIncoming(utils::get_integer_constant(0), current_bb);

				Value* branch_to_body = ir_builder->CreateICmpSLT(run_index_v, number_of_runs_v);
				ir_builder->CreateCondBr(branch_to_body, loop_body_bb, loop_end_bb);
			});

			// Abspeichern in welcher Iteration einer Schleife man sich aktuell befindet, um in einer
			// geschachtelten Klammer mit get_ptr_to_current_bracket den korrekten Zeiger auswählen zu
			// können
			iterations_through_repetitions.emplace_back(current_id, run_index_v);

			// values_over_all_runs zeigt auf denselben Kontext, egal welches Element aus
			// bracket_from_all_runs ausgewählt wird
			current_context = bracket_from_all_runs.at(0).first->values_over_all_runs;

			enclosing_function->getBasicBlockList().push_back(loop_body_bb);
			utils::generate_on_basic_block(loop_body_bb, [&]() {
				// Der von setup_selects generiert Code muss in jede Iteration der Klammer ausgeführt werden
				setup_selects(run_index_v);

#ifdef RUNTIME_DEBUG_OUTPUT
				std::vector<Value*> args1;
				args1.push_back(utils::construct_string("=========================== run: %d\n"));
				args1.push_back(run_index_v);
				ir_builder->CreateCall(predef_functions::printf(), args1);
#endif

				if(needs_switch) {
					Value* chosen_options_v = utils::extract_from_i(ptr_to_current_bracket, "", 0, 0);
					Value* chosen_option_v = utils::extract_from_v(chosen_options_v, "", run_index_v);
					SwitchInst* switch_inst = ir_builder->CreateSwitch(chosen_option_v, loop_end_bb, options.size());
					for(int i = 0; i < options.size(); ++i)
						switch_inst->addCase(utils::get_integer_constant(i), options.at(i).block);

					for(Option& option : options) {
						enclosing_function->getBasicBlockList().push_back(option.block);
						utils::generate_on_basic_block(option.block, [&]() {
							gen_expr_code_internal(option.expression);
							// gen_expr_code_internal könnte neue Blöcke erzeugt haben
							option.block = ir_builder->GetInsertBlock();
							ir_builder->CreateBr(loop_opt_merge);
						});
					}
				} else {
					gen_expr_code_internal(options.at(0).expression);
					ir_builder->CreateBr(loop_opt_merge);
				}

				enclosing_function->getBasicBlockList().push_back(loop_opt_merge);
				utils::generate_on_basic_block(loop_opt_merge, [&]() {
					Value* i_plus_one =
							ir_builder->CreateAdd(run_index_v, utils::get_integer_constant(1), "run.index.inc"_uniq);
					run_index_v->addIncoming(i_plus_one, loop_opt_merge);
					ir_builder->CreateBr(loop_begin_bb);
				});
			});

			iterations_through_repetitions.pop_back();

			enclosing_function->getBasicBlockList().push_back(loop_end_bb);
			ir_builder->SetInsertPoint(loop_end_bb);
			return_value = flx_value::promote_to_flx_value(number_of_runs_v, flx_value::nat);
			break;
		}
		case option: {
			setup_selects();

			BasicBlock* has_option_bb = BasicBlock::Create(llvm_module->getContext(), "opt.has.option"_uniq);
			BasicBlock* select_option_bb = BasicBlock::Create(llvm_module->getContext(), "opt.select.option"_uniq);
			BasicBlock* else_option_bb;
			BasicBlock* merge_options_bb = BasicBlock::Create(llvm_module->getContext(), "opt.merge.option"_uniq);
			bool has_else = expr(row_)[CH::Z + 1](passes_)[CH::A](choice_) == CH::A;
			if(has_else)
				else_option_bb = BasicBlock::Create(llvm_module->getContext(), "opt.else.option"_uniq);
			BasicBlock* else_or_merge = has_else ? else_option_bb : merge_options_bb;

			Value* has_option_v = utils::extract_from_i(ptr_to_current_bracket, "opt.has.opt"_uniq, 0, 2);
			Value* chosen_option_per_run_v = utils::extract_from_i(ptr_to_current_bracket, "", 0, 0);
			Value* chosen_option_v = utils::extract_from_i(chosen_option_per_run_v, "opt.chosen.opt"_uniq, 0);
			Value* chosen_option_plus_one_v = flx_value::promote_to_flx_value(
					ir_builder->CreateAdd(chosen_option_v, utils::get_integer_constant(1), "opt.chosen.opt.inc"_uniq),
					flx_value::nat);

			ir_builder->CreateBr(has_option_bb);
			enclosing_function->getBasicBlockList().push_back(has_option_bb);
			utils::generate_on_basic_block(has_option_bb, [&has_option_v, &select_option_bb, &else_or_merge]() {
				ir_builder->CreateCondBr(has_option_v, select_option_bb, else_or_merge);
			});

			enclosing_function->getBasicBlockList().push_back(select_option_bb);
			utils::generate_on_basic_block(select_option_bb, [&chosen_option_v, &else_or_merge, &options]() {
				SwitchInst* switch_inst = ir_builder->CreateSwitch(chosen_option_v, else_or_merge, options.size());
				for(int i = 0; i < options.size(); ++i)
					switch_inst->addCase(utils::get_integer_constant(i), options.at(i).block);
			});

			// values_over_all_runs zeigt auf denselben Kontext, egal welches Element aus
			// bracket_from_all_runs ausgewählt wird
			current_context = bracket_from_all_runs.at(0).first->values_over_all_runs;
			for(Option& option : options) {
				enclosing_function->getBasicBlockList().push_back(option.block);
				utils::generate_on_basic_block(option.block, [&option, &merge_options_bb]() {
					gen_expr_code_internal(option.expression);
					// gen_expr_code_internal könnte neue Blöcke erzeugt haben
					option.block = ir_builder->GetInsertBlock();
					ir_builder->CreateBr(merge_options_bb);
				});
			}

			if(has_else) {
				enclosing_function->getBasicBlockList().push_back(else_option_bb);
				utils::generate_on_basic_block(else_option_bb, [&]() {
					gen_expr_code_internal(expr(row_)[CH::Z + 1](passes_)[CH::A](branch_)[CH::Z](opnd_));
					else_option_bb = ir_builder->GetInsertBlock();
					ir_builder->CreateBr(merge_options_bb);
				});
			}

			enclosing_function->getBasicBlockList().push_back(merge_options_bb);
			utils::generate_on_basic_block(merge_options_bb, [&]() {
				assert(merge_options_bb->hasNPredecessors(options.size() + 1 + !has_else));
				PHINode* phi =
						ir_builder->CreatePHI(utils::get_integer_type(FLX_VALUE_WIDTH), options.size() + 1 + !has_else);
				if(!has_else)
					// Diese Möglichkeit sollte nie gewählt werden
					phi->addIncoming(flx_value::get(flx_value::nil), select_option_bb);
				for(const auto& [_, block] : options)
					phi->addIncoming(chosen_option_plus_one_v, block);
				if(has_else)
					phi->addIncoming(chosen_option_plus_one_v, else_option_bb);
				else
					phi->addIncoming(flx_value::get(flx_value::nil), has_option_bb);
				return_value = phi;
			});
			break;
		}
		default: assert_not_reached();
	}
	current_context = old_context;

	bracket_stack.pop_back();
	return return_value;
}

Value* assign_gen(Expr expr) {
	std::stringstream name;
	name << expr(row_)[CH::A + 1](opnd_)(row_)[CH::A](word_);
	Value* new_value = gen_expr_code_internal(expr(row_)[CH::Z](opnd_));
	ValueContext::ValueOrArgIndex* alloca = current_context->lookup_value(name.str());
	assert(alloca != nullptr && alloca->value != nullptr);
	ir_builder->CreateStore(new_value, alloca->value);
	return new_value;
}

Value* query_gen(Expr expr) {
	std::stringstream name;
	name << expr(row_)[CH::Z](opnd_)(row_)[CH::A](word_);
	ValueContext::ValueOrArgIndex* alloca = current_context->lookup_value(name.str());
	// Wenn in einer Impl-Klammer eine Variable verwendet wird die Klammer aber nicht durchlaufen wird,
	// wird diese hier nicht gefunden.
	// Parameter eckiger Klammern mit nur Alternative bekommen keinen eigenen Kontext und
	// sind auch außerhalb der Klammer sichtbar, weswegen auf sie im Gegensatz zu allen anderen Parametern während
	// der Laufzeit tatsächlich zugegriffen werden kann, auch wenn die Klammer nicht durchlaufen wurde. Deswegen
	// muss hier für genau diesen Fall der wohldefinierte Wert nil zurückgegeben werden.
	if(alloca == nullptr) {
#ifdef COMPILE_TIME_DEBUG_OUTPUT
		std::cout << "Variable " << name.str() << " nicht gefunden!" << std::endl;
#endif
		return flx_value::get(flx_value::nil);
	}
	return ir_builder->CreateLoad(utils::get_integer_type(FLX_VALUE_WIDTH), alloca->value, utils::mkuniq(name.str()));
}