idf_component_register(
    SRCS
        "main.cpp"
        "model_data.cc"

    INCLUDE_DIRS
        "."

    REQUIRES
        esp_timer
        esp_hw_support
)