
cmake_minimum_required(VERSION 3.10)

project(WindProject)

find_package(glew CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

add_executable(WindProject main.cpp)

target_link_libraries(WindProject
    PRIVATE
    GLEW::GLEW
    glfw
    glm::glm
)
