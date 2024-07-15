find_package(ament_cmake_gtest)

set(GTEST_BIN_NAME ${PROJECT_NAME}-gtest)

ament_add_gtest(${GTEST_BIN_NAME} test.cpp)
if(TARGET ${GTEST_BIN_NAME})
  target_link_libraries(${GTEST_BIN_NAME} ${PROJECT_NAME})
  install(
    TARGETS ${GTEST_BIN_NAME}
    RUNTIME DESTINATION lib/${PROJECT_NAME}
  )
endif()