#pragma once

enum class CompilerID { VisualStudio, GCC, Clang, Emscripten, Unknown };

class Compiler {

public:

	constexpr static CompilerID GetCompiler();

private:

#ifdef _MSC_VER

	constexpr static CompilerID compiler = CompilerID::VisualStudio;

#elif __GNUC__

	constexpr static CompilerID compiler = CompilerID::GCC;

#elif __clang__

	constexpr static CompilerID compiler = CompilerID::Clang;

#elif __EMSCRIPTEN__ 

	constexpr static CompilerID compiler = CompilerID::Emscripten;

#else

	constexpr static CompilerID compiler = CompilerID::Unknown;

#endif

};

constexpr CompilerID Compiler::GetCompiler() {
	return compiler;
}