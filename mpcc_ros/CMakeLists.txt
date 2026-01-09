cmake_minimum_required(VERSION 3.8)
project(mpcc_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(ackermann_msgs REQUIRED)
find_package(vesc_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Threads REQUIRED)
find_package(rclcpp_components REQUIRED)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

find_package(OpenMP REQUIRED)

find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackPoints.msg"
  "msg/SHint.msg"
  "msg/Corridor.msg"
  "msg/MpccControl.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)


set(MPCC_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../MPCC/C++)
set(HPIPM_LIB ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/hpipm/lib/lib/libhpipm.a)
set(BLASFEO_LIB ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/blasfeo/lib/lib/libblasfeo.a)


include_directories(
  include
  ${MPCC_SRC_DIR}
  ${MPCC_SRC_DIR}/Constraints
  ${MPCC_SRC_DIR}/Cost
  ${MPCC_SRC_DIR}/Interfaces
  ${MPCC_SRC_DIR}/MPC
  ${MPCC_SRC_DIR}/Model
  ${MPCC_SRC_DIR}/Params
  ${MPCC_SRC_DIR}/Spline
	
  ${MPCC_SRC_DIR}/External/blasfeo/include 
  ${MPCC_SRC_DIR}/External/blasfeo/lib/include
  ${MPCC_SRC_DIR}/External/hpipm/lib/include
  ${MPCC_SRC_DIR}/External/Eigen
  ${MPCC_SRC_DIR}/External/Json/include
  ${MPCC_SRC_DIR}/External/matplotlib
  ${Python3_INCLUDE_DIRS}
)

file(GLOB_RECURSE MPCC_SOURCES
  ${MPCC_SRC_DIR}/MPC/*.cpp
  ${MPCC_SRC_DIR}/Model/*.cpp
  ${MPCC_SRC_DIR}/Params/*.cpp
  ${MPCC_SRC_DIR}/Constraints/*.cpp
  ${MPCC_SRC_DIR}/Cost/*.cpp
  ${MPCC_SRC_DIR}/Spline/*.cpp
  ${MPCC_SRC_DIR}/Interfaces/*.cpp
  ${MPCC_SRC_DIR}/Plotting/*.cpp 
  ${MPCC_SRC_DIR}/types.cpp
)

add_executable(mpcc_node src/mpcc_node.cpp ${MPCC_SOURCES})
add_executable(mpcc_node_sim src/mpcc_node_sim.cpp ${MPCC_SOURCES})
add_executable(mpcc_node_sim_test src/mpcc_node_sim_test.cpp ${MPCC_SOURCES})
add_executable(mpcc_node_sim_delay_test src/mpcc_node_sim_delay_test.cpp ${MPCC_SOURCES})
add_executable(mpcc_main src/mpcc_main.cpp ${MPCC_SOURCES})



install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY launch/  DESTINATION share/${PROJECT_NAME}/launch)


add_executable(track_points_publisher
  src/track_points_publisher.cpp
  ${MPCC_SOURCES}  # MPCC 관련 소스 같이 빌드
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(track_points_publisher "${cpp_typesupport_target}")

target_link_libraries(
  mpcc_node
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/hpipm/lib/lib/libhpipm.a
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/blasfeo/lib/lib/libblasfeo.a
  Python3::Python
  m
  OpenMP::OpenMP_CXX
)

target_link_libraries(
  mpcc_node_sim
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/hpipm/lib/lib/libhpipm.a
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/blasfeo/lib/lib/libblasfeo.a
  Python3::Python
  m
  OpenMP::OpenMP_CXX
)

target_link_libraries(
  mpcc_node_sim_test
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/hpipm/lib/lib/libhpipm.a
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/blasfeo/lib/lib/libblasfeo.a
  Python3::Python
  m
  OpenMP::OpenMP_CXX
)

target_link_libraries(
  mpcc_node_sim_delay_test
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/hpipm/lib/lib/libhpipm.a
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/blasfeo/lib/lib/libblasfeo.a
  Python3::Python
  m
  OpenMP::OpenMP_CXX
)

target_link_libraries(
  track_points_publisher
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/hpipm/lib/lib/libhpipm.a
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/blasfeo/lib/lib/libblasfeo.a
  Python3::Python
  m
  OpenMP::OpenMP_CXX
)

target_link_libraries(
  mpcc_main
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/hpipm/lib/lib/libhpipm.a
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/blasfeo/lib/lib/libblasfeo.a
  Python3::Python
  m
  OpenMP::OpenMP_CXX
  Threads::Threads
)

target_include_directories(mpcc_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(mpcc_node PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fopenmp>)
target_compile_features(mpcc_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_include_directories(mpcc_node_sim PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(mpcc_node_sim PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fopenmp>)
target_compile_features(mpcc_node_sim PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_include_directories(mpcc_node_sim_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(mpcc_node_sim_test PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fopenmp>)
target_compile_features(mpcc_node_sim_test PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_include_directories(mpcc_node_sim_delay_test PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(mpcc_node_sim_delay_test PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fopenmp>)
target_compile_features(mpcc_node_sim_delay_test PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17


target_include_directories(track_points_publisher PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(track_points_publisher PUBLIC c_std_99 cxx_std_17)

target_include_directories(mpcc_main PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_options(mpcc_main PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-fopenmp>)
target_compile_features(mpcc_main PUBLIC c_std_99 cxx_std_17)


ament_target_dependencies(mpcc_node
  rclcpp
  nav_msgs
  geometry_msgs
  ackermann_msgs
  tf2_geometry_msgs
  vesc_msgs
  tf2
  tf2_ros
  Eigen3
  visualization_msgs
  std_msgs
  rosidl_default_runtime
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(mpcc_node "${cpp_typesupport_target}")

ament_target_dependencies(mpcc_node_sim
  rclcpp
  nav_msgs
  geometry_msgs
  ackermann_msgs
  tf2_geometry_msgs
  vesc_msgs
  tf2
  tf2_ros
  Eigen3
  visualization_msgs
  std_msgs
)

ament_target_dependencies(mpcc_node_sim_test
  rclcpp
  nav_msgs
  geometry_msgs
  ackermann_msgs
  tf2_geometry_msgs
  vesc_msgs
  tf2
  tf2_ros
  Eigen3
  visualization_msgs
  std_msgs
  rosidl_default_runtime
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(mpcc_node_sim_test "${cpp_typesupport_target}")


ament_target_dependencies(mpcc_node_sim_delay_test
  rclcpp
  nav_msgs
  geometry_msgs
  ackermann_msgs
  tf2_geometry_msgs
  vesc_msgs
  tf2
  tf2_ros
  Eigen3
  visualization_msgs
  std_msgs
)

ament_target_dependencies(track_points_publisher
  rclcpp
  geometry_msgs
  std_msgs
  visualization_msgs
  nav_msgs
  Eigen3
  rosidl_default_runtime  # 메시지 런타임 의존성
)

ament_target_dependencies(mpcc_main
  rclcpp
  nav_msgs
  geometry_msgs
  ackermann_msgs
  tf2_geometry_msgs
  vesc_msgs
  tf2
  tf2_ros
  Eigen3
  visualization_msgs
  std_msgs
)


install(TARGETS 
  mpcc_node
  mpcc_node_sim
  mpcc_node_sim_test
  mpcc_node_sim_delay_test
  track_points_publisher
  mpcc_main 
  DESTINATION lib/${PROJECT_NAME})

# 새 노드 추가
add_executable(mpcc_node_sim_test_planner
  src/mpcc_node_sim_test_planner.cpp   # 새 파일 이름
  ${MPCC_SOURCES}
)

# ROS 2 메시지 타입 지원 추가
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(mpcc_node_sim_test_planner "${cpp_typesupport_target}")

# 링킹
target_link_libraries(
  mpcc_node_sim_test_planner
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/hpipm/lib/lib/libhpipm.a
  ${CMAKE_SOURCE_DIR}/../MPCC/C++/External/blasfeo/lib/lib/libblasfeo.a
  Python3::Python
  m
  OpenMP::OpenMP_CXX
)

# include 및 C++ 표준
target_include_directories(mpcc_node_sim_test_planner PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(mpcc_node_sim_test_planner PUBLIC c_std_99 cxx_std_17)

# 의존성
ament_target_dependencies(mpcc_node_sim_test_planner
  rclcpp
  nav_msgs
  geometry_msgs
  ackermann_msgs
  tf2_geometry_msgs
  vesc_msgs
  tf2
  tf2_ros
  Eigen3
  visualization_msgs
  std_msgs
  rosidl_default_runtime  # TrackPoints.msg 사용 위해 필요
)

# 설치
install(TARGETS
  mpcc_node_sim_test_planner
  DESTINATION lib/${PROJECT_NAME})









# ----------------------------------------------------------------------
# components (SHARED)
# ----------------------------------------------------------------------
add_library(mpcc_components SHARED
  src/mpcc_components.cpp
  src/shared_state.cpp
  ${MPCC_SOURCES}
)
# 공유 라이브러리 + 정적 아카이브 조합 안전하게
set_target_properties(mpcc_components PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(mpcc_components PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(mpcc_components
  rclcpp
  rclcpp_components
  tf2 tf2_geometry_msgs tf2_ros
  geometry_msgs visualization_msgs nav_msgs ackermann_msgs
  std_msgs Eigen3 vesc_msgs
)

# 🔑 정적 의존성은 전이되지 않으니, 컴포넌트에 직접 링크
target_link_libraries(mpcc_components
  ${HPIPM_LIB}
  ${BLASFEO_LIB}
  Python3::Python
  m
  OpenMP::OpenMP_CXX
)

# 컴포넌트 등록 (클래스 이름과 일치)
rclcpp_components_register_nodes(mpcc_components
  "MPCCSim" "MPCCOne" "MPCCTwo"
)

install(TARGETS mpcc_components
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

add_executable(corridor_generator src/corridor_generator.cpp)
ament_target_dependencies(corridor_generator
  rclcpp
  std_msgs
  # 메시지 런타임 (SHint.msg, Corridor.msg)
  rosidl_default_runtime
)

ament_target_dependencies(corridor_generator
  rclcpp
  std_msgs
  rosidl_default_runtime
)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(corridor_generator "${cpp_typesupport_target}")


target_include_directories(corridor_generator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_features(corridor_generator PUBLIC c_std_99 cxx_std_17)

install(TARGETS corridor_generator
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
