#pragma once

#include <functional>

#include "data.h"
#include "seq.ch"

namespace llvm {
class Module;
class Value;
}

enum SettingsFlags {
	O0 = 1 << 0,
	O1 = 1 << 1,
	O2 = 1 << 2,
	READABLE_FUNCTION_NAMES = 1 << 3,
	EXPAND_BRACKETS = 1 << 4
};

using CodeGenFunction = llvm::Value* (*)(Expr);

ATTR1(code_gen_, Oper, CodeGenFunction)

int gen_expr_code(Expr expr,
		int flags,
		std::function<int(llvm::Module&)> pre_code_gen,
		std::function<int(llvm::Module&)> post_code_gen);
llvm::Value* gen_expr_code_internal(Expr expr);
llvm::Value* sequ_gen(Expr expr);
llvm::Value* print_gen(Expr expr);
llvm::Value* cdecl_gen(Expr expr);
llvm::Value* logic_gen(Expr expr);
llvm::Value* neg_gen(Expr expr);
llvm::Value* branch_gen(Expr expr);
llvm::Value* loop_gen(Expr expr);
llvm::Value* read_gen(Expr expr);
llvm::Value* cmp_gen(Expr expr);
llvm::Value* bin_gen(Expr expr);
llvm::Value* chs_gen(Expr expr);
llvm::Value* fac_gen(Expr expr);
llvm::Value* paren_gen(Expr expr);
llvm::Value* intlit_gen(Expr expr);
llvm::Value* boollit_gen(Expr expr);
llvm::Value* const_gen(Expr expr);
llvm::Value* odecl_gen(Expr expr);
llvm::Value* appl_gen(Expr expr);
llvm::Value* bracket_gen(Expr expr);
llvm::Value* assign_gen(Expr expr);
llvm::Value* query_gen(Expr expr);