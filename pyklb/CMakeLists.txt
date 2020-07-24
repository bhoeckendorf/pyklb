add_cython_target(_pyklb C)
add_library(_pyklb MODULE ${_pyklb})
target_link_libraries(_pyklb klb_static)
python_extension_module(_pyklb)

install(TARGETS _pyklb LIBRARY DESTINATION pyklb)
