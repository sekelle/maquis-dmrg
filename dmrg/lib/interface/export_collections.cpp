#include <memory>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace boost::python;

/** @brief Type that allows for registration of conversions from
 *         Python iterable types.
 */
struct iterable_converter
{
    /** @note Registers converter from a Python iterable type to the
     *  provided type.
     */
    template<typename Container>
    iterable_converter&
    from_python()
    {
        boost::python::converter::registry::push_back(&iterable_converter::convertible,
                                                      &iterable_converter::construct<Container>,
                                                      boost::python::type_id<Container>());

        // support chaining
        return *this;
    }

    /// @brief Check if PyObject is iterable
    static void* convertible(PyObject* object)
    {
        //return PyObject_GetIter(object) ? object : nullptr;
        return PyObject_GetIter(object) ? object : NULL;
    }

    /** @brief Convert iterable PyObject to C++ container type.
     *
     * Container concept requirements:
     *
     *   * Container::value_type is CopyConstructable.
     *   * Container can be constructed and populated with two iterators.
     *     i.e. Container(begin, end)
     */
    template<typename Container>
    static void construct(PyObject* object,
                          boost::python::converter::rvalue_from_python_stage1_data* data)
    {
        namespace python = boost::python;

        // Object is borrowed reference, so create a handle indictating it is
        // borrowed for proper reference counting
        python::handle<> handle(python::borrowed(object));

        // Obtain a handle to the memory block that the converter has allocated
        // for the C++ type.
        typedef python::converter::rvalue_from_python_storage<Container> storage_type;

        void* storage = reinterpret_cast<storage_type*>(data)->storage.bytes;

        typedef python::stl_input_iterator<typename Container::value_type> iterator;

        // Allocate the C++ type into the converter's memory block, and assign
        // its handle to the converter's convertible variable. The C++
        // container is populated by passing the begin and end iterators of
        // the python object to the container's constructor.
        new (storage) Container(iterator(python::object(handle)), // begin
                                iterator());                      // end
        data->convertible = storage;
    }
};

void export_collections()
{
    // => Iterable Conversions <= //
    
    iterable_converter()
        .from_python<std::vector<size_t> >()
        .from_python<std::vector<int> >()
        .from_python<std::vector<double> >()
        .from_python<std::vector<bool> >()
        .from_python<std::vector<std::vector<int> > >()
        .from_python<std::vector<std::vector<size_t> > >()
        .from_python<std::vector<std::vector<double> > >()
        .from_python<std::vector<std::string> >()
        ;

    // => Standard Collections <= //

    class_<std::vector<bool> >("BoolVec")
        .def(vector_indexing_suite<std::vector<bool> >())
        ;

    class_<std::vector<double> >("DoubleVec")
        .def(vector_indexing_suite<std::vector<double> >())
        ;

    class_<std::vector<std::vector<double> > >("DoubleVecVec")
        .def(vector_indexing_suite<std::vector<std::vector<double> > >())
        ;

    class_<std::vector<int> >("IntVec")
        .def(vector_indexing_suite<std::vector<int> >())
        ;

    class_<std::vector<std::vector<int> > >("IntVecVec")
        .def(vector_indexing_suite<std::vector<std::vector<int> > >())
        ;

    class_<std::vector<std::string> >("StringVec")
        .def(vector_indexing_suite<std::vector<std::string> >())
        ;

    class_<std::vector<size_t> >("Size_tVec")
        .def(vector_indexing_suite<std::vector<size_t> >())
        ;

    class_<std::vector<std::vector<size_t> > >("Size_tVecVec")
        .def(vector_indexing_suite<std::vector<std::vector<size_t> > >())
        ;

    class_<std::pair<int, int> >("IntPair")
        .def_readwrite("first", &std::pair<int,int>::first)
        .def_readwrite("second", &std::pair<int,int>::second)
        ;

    class_<std::vector<std::pair<int,int> > >("IntPairVec")
        .def(vector_indexing_suite<std::vector<std::pair<int,int> > >())
        ;

    class_<std::vector<std::vector<std::pair<int,int> > > >("IntPairVecVec")
        .def(vector_indexing_suite<std::vector<std::vector<std::pair<int,int> > > >())
        ;
}
