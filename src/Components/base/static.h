#ifndef STATIC_H
#define STATIC_H

namespace MLT
{
namespace Base
{
	struct Itself {};
	
	template <class Exact>
	class Any 
	{};

	template <typename Exact>
	Exact& exact(Any<Exact>& ref)
	{
		return *(Exact*)(void*)(&ref);
	}

	template <typename Exact>
	const Exact& exact(const Any<Exact>& cref)
	{
		return *(const Exact*)(const void*)(&cref);
	}

	template <typename Exact>
	Exact* exact(Any<Exact>* ptr)
	{
		return (Exact*)(void*)(ptr);
	}

	template <typename Exact>
	const Exact* exact(const Any<Exact>* cptr)
	{
		return (const Exact*)(const void*)(cptr);
	}
	
	template <class T, class Exact>
	struct FindExact
	{
		typedef Exact ret;
	};
	
	template <class T>
	struct FindExact<T, Itself>
	{
		typedef T ret;
	};	

	#define FIND_EXACT(Type) typename FindExact<Type<Exact>, Exact>::ret

	#define FIND_EXACT_WITH_PARAMS(Type, Params) typename FindExact<Type<Params, Exact>, Exact>::ret
}
}
#endif // STATIC_H

/*

// class : class Class1
#define MLT_CLASS(Class) \
template <typename Params = MLT::Base::_Params, typename Exact = MLT::Base::Itself> \
class Class : public MLT::Base::Any<Exact>

// class derived of parent: class Class1 : Parent1
#define MLT_DER_CLASS(Class, Parent) \
template <typename Params = MLT::Base::_Params, typename Exact = MLT::Base::Itself> \
class Class : public Parent<Params, typename MLT::Base::FindExact<Class<Params, Exact>, Exact>::ret>

#define MLT_PARENT(Class, Parent) Parent<Params, typename MLT::Base::FindExact<Class<Params, Exact>, Exact>::ret> 

*/