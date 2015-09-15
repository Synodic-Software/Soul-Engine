#pragma once

#include "Engine Core/BasicDependencies.h"


class KeyFunction
{
public:
	virtual void Execute() const = 0;
};

template <class FunctionInstance>
class InputHelper : public KeyFunction
{
public:
	InputHelper()    // constructor
	{
		function = 0;
	}

	typedef void (FunctionInstance::*FunctionPointer)(int Param);

	virtual void Execute(int Param) const
	{
		if (function) (cInst->*function)(Param);
	}

	void SetCallback(FunctionInstance  *cInstancePointer,
		FunctionPointer   pFunctionPointer)
	{
		cInst = cInstancePointer;
		function = pFunctionPointer;
	}

private:
	FunctionInstance  *cInst;
	FunctionPointer  function;
};

