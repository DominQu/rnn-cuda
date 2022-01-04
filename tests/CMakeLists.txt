add_executable(multiplication matrix.cpp)

target_link_libraries(multiplication rnn-lib)

enable_testing()

add_test(multiplication multiplication)
set_tests_properties(multiplication PROPERTIES FIXTURES_REQUIRED test_fixture)
