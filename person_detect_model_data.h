#ifndef PERSON_DETECT_MODEL_DATA_H_
#define PERSON_DETECT_MODEL_DATA_H_

#include <cstdint>

// Declare external pointers to model data
extern const unsigned char g_person_detect_model_data1[];
extern const unsigned char g_person_detect_model_data2[];
extern const unsigned char g_person_detect_model_data3[];

// Add array of pointers for easy access in main_functions
static const unsigned char* g_person_detect_model_data_array[] = {
    g_person_detect_model_data1,
    g_person_detect_model_data2,
    g_person_detect_model_data3
};

#endif  // PERSON_DETECT_MODEL_DATA_H_

