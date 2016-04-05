#ifndef MLT_MODELS_PIPELINE_HPP
#define MLT_MODELS_PIPELINE_HPP

#include <Eigen/Core>

#include "transformers/transformer_mixin.hpp"

namespace mlt {
namespace models {

	// Forward declaration
	template <class... Ts>
	class Pipeline {
	};

	template<size_t _Index, class _Pipline>
	struct sub_pipeline_element;

	template<size_t _Index>
	struct sub_pipeline_element<_Index, Pipeline<> >
	{
		static_assert(std::_Always_false<std::integral_constant<size_t, _Index> >::value, "pipeline index out of bounds");
	};

	template<class _This, class... _Rest>
	struct sub_pipeline_element<0, Pipeline<_This, _Rest...> >
	{
		typedef _This model_type;
		typedef Pipeline<_This, _Rest...> type;
	};

	template<size_t _Index, class _This, class... _Rest>
	struct sub_pipeline_element<_Index, Pipeline<_This, _Rest...> >
		: public sub_pipeline_element<_Index - 1, Pipeline<_Rest...> >
	{};

	template<size_t _Index, class _Pipline>
	struct sub_pipeline_element<_Index, const _Pipline>
		: public sub_pipeline_element<_Index, _Pipline>
	{
		typedef typename std::add_const<typename sub_pipeline_element<_Index, _Pipline>::model_type>::type model_type;
		typedef typename std::add_const<typename sub_pipeline_element<_Index, _Pipline>::type>::type type;
	};

	template<size_t _Index, class _Pipline>
	struct sub_pipeline_element<_Index, volatile _Pipline> : public sub_pipeline_element<_Index, _Pipline>
	{
		typedef typename std::add_volatile<typename sub_pipeline_element<_Index, _Pipline>::model_type>::type model_type;
		typedef typename std::add_volatile<typename sub_pipeline_element<_Index, _Pipline>::type>::type type;
	};

	template<size_t _Index, class _Pipline>
	struct sub_pipeline_element<_Index, const volatile _Pipline> : public sub_pipeline_element<_Index, _Pipline>
	{
		typedef typename std::add_cv<typename sub_pipeline_element<_Index, _Pipline>::model_type>::type model_type;
		typedef typename std::add_cv<typename sub_pipeline_element<_Index, _Pipline>::type>::type type;
	};

	template<>
	class Pipeline<> {
		Pipeline() = delete;
	};

	template <class T>
	class Pipeline<T> {
	public:
		explicit Pipeline(T& estimator) : _estimator(estimator) {}

		using pipeline_t = typename Pipeline<T>;

		template<size_t _Index>
		using pipeline_model_t = typename sub_pipeline_element<_Index, pipeline_t>::model_type;

		template<size_t _Index>
		using sub_pipeline_t = typename sub_pipeline_element<_Index, pipeline_t>::type;

		Pipeline<T>& fit(const Eigen::MatrixXd& input)
		{
			_estimator.fit(input);
			return *this;
		}

		Pipeline<T>& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target)
		{
			_estimator.fit(input, target);
			return *this;
		}

		Pipeline<T>& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXi& classes)
		{
			_estimator.fit(input, classes);
			return *this;
		}

		Eigen::MatrixXd transform(const Eigen::MatrixXd& input) const
		{
			return _estimator.transform(input);
		}

		Eigen::MatrixXi classify(const Eigen::MatrixXd& input) const
		{
			return _estimator.classify(input);
		}

		Eigen::MatrixXd predict(const Eigen::MatrixXd& input) const
		{
			return _estimator.predict(input);
		}

		Eigen::VectorXi cluster(const Eigen::MatrixXd& input) const
		{
			return _estimator.cluster(input);
		}

		T& top_estimator() { return _estimator; }

		template<size_t Index>
		inline pipeline_t& get()
		{
			typedef sub_pipeline_t<Index> _;
			return *this;
		}

		template<size_t Index>
		inline T& get_model()
		{
			typedef sub_pipeline_t<Index> _;
			return _estimator;
		}

	protected:
		T& _estimator;
	};

	template <class T, class... Ts>
	class Pipeline<T, Ts...> : protected Pipeline<Ts...>, public transformers::TransformerMixin<Pipeline<T, Ts...>> {
	public:
		explicit Pipeline(T& estimator, Ts & ... following) : Pipeline<Ts...>(following...), _estimator(estimator) {}

		using pipeline_t = typename Pipeline<T, Ts...>;

		template<size_t _Index>
		using pipeline_model_t = typename sub_pipeline_element<_Index, pipeline_t>::model_type;

		template<size_t _Index>
		using sub_pipeline_t = typename sub_pipeline_element<_Index, pipeline_t>::type;

		pipeline_t& fit(const Eigen::MatrixXd& input)
		{
			sub_pipeline_t<1>::fit(_estimator.fit_transform(input));
			return *this;
		}

		pipeline_t& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target)
		{
			sub_pipeline_t<1>::fit(_estimator.fit_transform(input, target), target);
			return *this;
		}

		pipeline_t& fit(const Eigen::MatrixXd& input, const Eigen::MatrixXi& classes)
		{
			sub_pipeline_t<1>::fit(_estimator.fit_transform(input, classes), classes);
			return *this;
		}

		Eigen::MatrixXd transform(const Eigen::MatrixXd& input) const
		{
			return sub_pipeline_t<1>::transform(_estimator.transform(input));
		}

		Eigen::MatrixXi classify(const Eigen::MatrixXd& input) const
		{
			return sub_pipeline_t<1>::classify(_estimator.transform(input));
		}

		Eigen::MatrixXd predict(const Eigen::MatrixXd& input) const
		{
			return sub_pipeline_t<1>::predict(_estimator.transform(input));
		}

		Eigen::MatrixXi cluster(const Eigen::MatrixXd& input) const
		{
			return sub_pipeline_t<1>::cluster(_estimator.transform(input));
		}

		T& top_estimator() { return _estimator; }

		template<size_t Index>
		inline sub_pipeline_t<Index>& get()
		{
			return static_cast<sub_pipeline_t<Index>&>(*this);
		}

		template<size_t Index>
		inline typename pipeline_model_t<Index>& get_model()
		{
			return static_cast<sub_pipeline_t<Index>&>(*this).top_estimator();
		}

		//template<size_t _Index>
		//inline constexpr const typename sub_pipeline_element<_Index, Pipeline<T, Ts...> >::type&
		//get() const
		//{	// get const reference to _Index subpipeline of pipeline
		//	typedef typename sub_pipeline_element<_Index, Pipeline<_Types...> >::type _Ptype;
		//	return static_cast<_Ptype>(&*this);
		//}

		/*template <size_t k, class... Ts>
		typename std::enable_if<
			k == 0, typename pipiline_type_holder<0, Pipeline<Ts...>>::type&>::type
			get() {
			return *this;
		}

		template <size_t k, class T, class... Ts>
		typename std::enable_if<
			k != 0, typename elem_type_holder<k, tuple<T, Ts...>>::type&>::type
			get(tuple<T, Ts...>& t) {
			tuple<Ts...>& base = t;
			return get<k - 1>(base);
		}*/

	protected:
		T& _estimator;
	};

	template <class... Ts>
	Pipeline<Ts...> create_pipeline(Ts & ... models) {
		return Pipeline<Ts...>(models...);
	}
}
}
#endif