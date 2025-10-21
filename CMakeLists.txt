cmake_minimum_required(VERSION 3.20)
project(cf_proj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CheckCXXCompilerFlag)

function(add_flag_if_supported flag)
  string(MAKE_C_IDENTIFIER "${flag}" VAR)
  check_cxx_compiler_flag("${flag}" ${VAR})
  if(${VAR})
    add_compile_options("${flag}")
  endif()
endfunction()

function(add_dbg_flag_if_supported flag)
  string(MAKE_C_IDENTIFIER "${flag}" VAR)
  check_cxx_compiler_flag("${flag}" ${VAR})
  if(${VAR})
    add_compile_options("$<$<CONFIG:Debug>:${flag}>")
    add_link_options("$<$<CONFIG:Debug>:${flag}>")
  endif()
endfunction()

add_flag_if_supported("-fdiagnostics-color=always")
add_flag_if_supported("-fcolor-diagnostics")

add_flag_if_supported("-Wall")
add_flag_if_supported("-Wextra")
add_flag_if_supported("-Wshadow")

# Debug-only runtime checking
add_dbg_flag_if_supported("-O1")
add_dbg_flag_if_supported("-g")
add_dbg_flag_if_supported("-fno-omit-frame-pointer")
add_dbg_flag_if_supported("-fsanitize=address")

# Checked STL in Debug (bounds checks, etc.)
# Use OR for multiple compiler IDs, and emit separate -D tokens.
add_compile_definitions(
  $<$<AND:$<CONFIG:Debug>,$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>>:_GLIBCXX_DEBUG>
  $<$<AND:$<CONFIG:Debug>,$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:Clang>>>:_LIBCPP_DEBUG=1>
  $<$<AND:$<CONFIG:Debug>,$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:Clang>>>:_LIBCPP_ENABLE_ASSERTIONS=1>
)

add_executable(app submit.cpp)
