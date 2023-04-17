#ifndef GRAPH_INF_PYWRAPPER_INIT_UTILITY_MAPS_H
#define GRAPH_INF_PYWRAPPER_INIT_UTILITY_MAPS_H

#include <nanobind/nanobind.h>
#include <nanobind/stl/list.h>

#include "GraphInf/utility/maps.hpp"

namespace nb = nanobind;
namespace GraphInf
{
    template <typename KeyType, typename ValueType>
    nb::class_<Map<KeyType, ValueType>> declareMap(nb::module_ &m, std::string pyName)
    {
        return nb::class_<Map<KeyType, ValueType>>(m, pyName.c_str())
            .def(nb::init<const std::vector<KeyType> &, const std::vector<ValueType> &, ValueType>(), nb::arg("keys"), nb::arg("values"), nb::arg("default"))
            .def(nb::init<const Map<KeyType, ValueType> &>(), nb::arg("map"))
            .def(nb::init<ValueType>(), nb::arg("default"))
            .def("__getitem__", [](const Map<KeyType, ValueType> &self, KeyType key)
                 { return self.get(key); })
            .def("__eq__", [](Map<KeyType, ValueType> &self, Map<KeyType, ValueType> &other)
                 { return self.operator==(other); })
            .def("size", &Map<KeyType, ValueType>::size)
            .def("get", &Map<KeyType, ValueType>::get, nb::arg("key"))
            .def("set", &Map<KeyType, ValueType>::set, nb::arg("key"), nb::arg("value"))
            .def("is_empty", &Map<KeyType, ValueType>::isEmpty, nb::arg("key"))
            .def("erase", &Map<KeyType, ValueType>::erase, nb::arg("key"))
            .def("clear", &Map<KeyType, ValueType>::clear)
            // .def("display", &Map<KeyType, ValueType>::display)
            .def("get_keys", &Map<KeyType, ValueType>::getKeys)
            .def("get_values", &Map<KeyType, ValueType>::getValues);
    }

    template <typename KeyType>
    nb::class_<IntMap<KeyType>, Map<KeyType, int>> declareIntMap(nb::module_ &m, std::string pyName)
    {
        return nb::class_<IntMap<KeyType>, Map<KeyType, int>>(m, pyName.c_str())
            .def(nb::init<const std::vector<KeyType> &, const std::vector<int> &, int>(), nb::arg("keys"), nb::arg("values"), nb::arg("default") = 0)
            .def(nb::init<const IntMap<KeyType> &>(), nb::arg("map"))
            .def(nb::init<int>(), nb::arg("default") = 0)
            .def("increment", &IntMap<KeyType>::increment, nb::arg("key"), nb::arg("inc") = 1)
            .def("decrement", &IntMap<KeyType>::decrement, nb::arg("key"), nb::arg("dec") = 1);
    }

    template <typename KeyType>
    nb::class_<CounterMap<KeyType>, Map<KeyType, size_t>> declareCounterMap(nb::module_ &m, std::string pyName)
    {
        return nb::class_<CounterMap<KeyType>, Map<KeyType, size_t>>(m, pyName.c_str())
            .def(nb::init<const std::vector<KeyType> &, const std::vector<size_t> &, size_t>(), nb::arg("keys"), nb::arg("values"), nb::arg("default") = 0)
            .def(nb::init<const CounterMap<KeyType> &>(), nb::arg("map"))
            .def(nb::init<size_t>(), nb::arg("default") = 0)
            .def("increment", &CounterMap<KeyType>::increment, nb::arg("key"), nb::arg("inc") = 1)
            .def("decrement", &CounterMap<KeyType>::decrement, nb::arg("key"), nb::arg("dec") = 1)
            .def("get_sum", &CounterMap<KeyType>::getSum);
    }

    void initMaps(nb::module_ &m)
    {
        declareMap<size_t, int>(m, "UnIntKeyedIntValuedMap");
        declareIntMap<size_t>(m, "UnIntKeyedIntMap");

        declareMap<size_t, size_t>(m, "UnIntKeyedUnIntValuedMap");
        declareCounterMap<size_t>(m, "UnIntKeyedCounterMap");

        declareMap<std::pair<size_t, size_t>, size_t>(m, "UnIntPairKeyedUnIntValuedMap");
        declareCounterMap<std::pair<size_t, size_t>>(m, "UnIntPairKeyedCounterMap");
    }

}

#endif
