add_executable(test test.cpp matrix.cpp layers.cpp loader.cpp)

target_link_libraries(test rnn-linalg rnn-layers rnn-loader)

enable_testing()

add_test(test test)
set_tests_properties(test PROPERTIES FIXTURES_REQUIRED test_fixture)
