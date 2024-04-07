#include <pybind11/pybind11.h>

#include "razerz_gpu.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{

    m.def("razer_gpu_symmetric", &razer_gpu_symmetric);
    m.def("razer_gpu_symmetric_512", &razer_gpu_symmetric_512);
    m.def("razer_gpu_symmetric_256", &razer_gpu_symmetric_256);

    m.def("razer_gpu_symmetric_perf", &razer_gpu_symmetric_perf);
    m.def("razer_gpu_symmetric_512_perf", &razer_gpu_symmetric_512_perf);
    m.def("razer_gpu_symmetric_256_perf", &razer_gpu_symmetric_256_perf);

}

