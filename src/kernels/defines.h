#ifndef my_defines_vk // pragma once
#define my_defines_vk

#define GROUP_SIZE   1024
#define GROUP_SIZE_REDUX 32

#define RASSERT_ENABLED 0 // disabled by default, enable for debug by changing 0 to 1, disable before performance evaluation/profiling/commiting

#define WIDTH 2
#define BUCKETS (1u << WIDTH)

#endif // pragma once
