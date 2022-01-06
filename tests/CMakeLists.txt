add_executable(matrix matrix.cpp)

target_link_libraries(matrix rnn-lib)

enable_testing()

add_test(matrix matrix)
set_tests_properties(matrix PROPERTIES FIXTURES_REQUIRED test_fixture)
