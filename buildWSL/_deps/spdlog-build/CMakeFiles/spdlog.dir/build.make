# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/d/Projects/VirtualFluids_Develop

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/Projects/VirtualFluids_Develop/buildWSL

# Include any dependencies generated for this target.
include _deps/spdlog-build/CMakeFiles/spdlog.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include _deps/spdlog-build/CMakeFiles/spdlog.dir/compiler_depend.make

# Include the progress variables for this target.
include _deps/spdlog-build/CMakeFiles/spdlog.dir/progress.make

# Include the compile flags for this target's objects.
include _deps/spdlog-build/CMakeFiles/spdlog.dir/flags.make

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/spdlog.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/flags.make
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/spdlog.cpp.o: _deps/spdlog-src/src/spdlog.cpp
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/spdlog.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/VirtualFluids_Develop/buildWSL/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object _deps/spdlog-build/CMakeFiles/spdlog.dir/src/spdlog.cpp.o"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/spdlog-build/CMakeFiles/spdlog.dir/src/spdlog.cpp.o -MF CMakeFiles/spdlog.dir/src/spdlog.cpp.o.d -o CMakeFiles/spdlog.dir/src/spdlog.cpp.o -c /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/spdlog.cpp

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/spdlog.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/spdlog.cpp.i"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/spdlog.cpp > CMakeFiles/spdlog.dir/src/spdlog.cpp.i

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/spdlog.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/spdlog.cpp.s"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/spdlog.cpp -o CMakeFiles/spdlog.dir/src/spdlog.cpp.s

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/flags.make
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o: _deps/spdlog-src/src/stdout_sinks.cpp
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/VirtualFluids_Develop/buildWSL/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object _deps/spdlog-build/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/spdlog-build/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o -MF CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o.d -o CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o -c /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/stdout_sinks.cpp

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.i"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/stdout_sinks.cpp > CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.i

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.s"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/stdout_sinks.cpp -o CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.s

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/flags.make
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o: _deps/spdlog-src/src/color_sinks.cpp
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/VirtualFluids_Develop/buildWSL/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object _deps/spdlog-build/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/spdlog-build/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o -MF CMakeFiles/spdlog.dir/src/color_sinks.cpp.o.d -o CMakeFiles/spdlog.dir/src/color_sinks.cpp.o -c /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/color_sinks.cpp

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/color_sinks.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/color_sinks.cpp.i"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/color_sinks.cpp > CMakeFiles/spdlog.dir/src/color_sinks.cpp.i

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/color_sinks.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/color_sinks.cpp.s"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/color_sinks.cpp -o CMakeFiles/spdlog.dir/src/color_sinks.cpp.s

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/flags.make
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o: _deps/spdlog-src/src/file_sinks.cpp
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/VirtualFluids_Develop/buildWSL/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object _deps/spdlog-build/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/spdlog-build/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o -MF CMakeFiles/spdlog.dir/src/file_sinks.cpp.o.d -o CMakeFiles/spdlog.dir/src/file_sinks.cpp.o -c /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/file_sinks.cpp

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/file_sinks.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/file_sinks.cpp.i"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/file_sinks.cpp > CMakeFiles/spdlog.dir/src/file_sinks.cpp.i

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/file_sinks.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/file_sinks.cpp.s"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/file_sinks.cpp -o CMakeFiles/spdlog.dir/src/file_sinks.cpp.s

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/async.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/flags.make
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/async.cpp.o: _deps/spdlog-src/src/async.cpp
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/async.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/VirtualFluids_Develop/buildWSL/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object _deps/spdlog-build/CMakeFiles/spdlog.dir/src/async.cpp.o"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/spdlog-build/CMakeFiles/spdlog.dir/src/async.cpp.o -MF CMakeFiles/spdlog.dir/src/async.cpp.o.d -o CMakeFiles/spdlog.dir/src/async.cpp.o -c /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/async.cpp

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/async.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/async.cpp.i"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/async.cpp > CMakeFiles/spdlog.dir/src/async.cpp.i

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/async.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/async.cpp.s"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/async.cpp -o CMakeFiles/spdlog.dir/src/async.cpp.s

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/cfg.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/flags.make
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/cfg.cpp.o: _deps/spdlog-src/src/cfg.cpp
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/cfg.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/VirtualFluids_Develop/buildWSL/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object _deps/spdlog-build/CMakeFiles/spdlog.dir/src/cfg.cpp.o"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/spdlog-build/CMakeFiles/spdlog.dir/src/cfg.cpp.o -MF CMakeFiles/spdlog.dir/src/cfg.cpp.o.d -o CMakeFiles/spdlog.dir/src/cfg.cpp.o -c /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/cfg.cpp

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/cfg.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/cfg.cpp.i"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/cfg.cpp > CMakeFiles/spdlog.dir/src/cfg.cpp.i

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/cfg.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/cfg.cpp.s"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/cfg.cpp -o CMakeFiles/spdlog.dir/src/cfg.cpp.s

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/fmt.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/flags.make
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/fmt.cpp.o: _deps/spdlog-src/src/fmt.cpp
_deps/spdlog-build/CMakeFiles/spdlog.dir/src/fmt.cpp.o: _deps/spdlog-build/CMakeFiles/spdlog.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/Projects/VirtualFluids_Develop/buildWSL/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object _deps/spdlog-build/CMakeFiles/spdlog.dir/src/fmt.cpp.o"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT _deps/spdlog-build/CMakeFiles/spdlog.dir/src/fmt.cpp.o -MF CMakeFiles/spdlog.dir/src/fmt.cpp.o.d -o CMakeFiles/spdlog.dir/src/fmt.cpp.o -c /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/fmt.cpp

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/fmt.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/spdlog.dir/src/fmt.cpp.i"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/fmt.cpp > CMakeFiles/spdlog.dir/src/fmt.cpp.i

_deps/spdlog-build/CMakeFiles/spdlog.dir/src/fmt.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/spdlog.dir/src/fmt.cpp.s"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src/src/fmt.cpp -o CMakeFiles/spdlog.dir/src/fmt.cpp.s

# Object files for target spdlog
spdlog_OBJECTS = \
"CMakeFiles/spdlog.dir/src/spdlog.cpp.o" \
"CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o" \
"CMakeFiles/spdlog.dir/src/color_sinks.cpp.o" \
"CMakeFiles/spdlog.dir/src/file_sinks.cpp.o" \
"CMakeFiles/spdlog.dir/src/async.cpp.o" \
"CMakeFiles/spdlog.dir/src/cfg.cpp.o" \
"CMakeFiles/spdlog.dir/src/fmt.cpp.o"

# External object files for target spdlog
spdlog_EXTERNAL_OBJECTS =

_deps/spdlog-build/libspdlog.a: _deps/spdlog-build/CMakeFiles/spdlog.dir/src/spdlog.cpp.o
_deps/spdlog-build/libspdlog.a: _deps/spdlog-build/CMakeFiles/spdlog.dir/src/stdout_sinks.cpp.o
_deps/spdlog-build/libspdlog.a: _deps/spdlog-build/CMakeFiles/spdlog.dir/src/color_sinks.cpp.o
_deps/spdlog-build/libspdlog.a: _deps/spdlog-build/CMakeFiles/spdlog.dir/src/file_sinks.cpp.o
_deps/spdlog-build/libspdlog.a: _deps/spdlog-build/CMakeFiles/spdlog.dir/src/async.cpp.o
_deps/spdlog-build/libspdlog.a: _deps/spdlog-build/CMakeFiles/spdlog.dir/src/cfg.cpp.o
_deps/spdlog-build/libspdlog.a: _deps/spdlog-build/CMakeFiles/spdlog.dir/src/fmt.cpp.o
_deps/spdlog-build/libspdlog.a: _deps/spdlog-build/CMakeFiles/spdlog.dir/build.make
_deps/spdlog-build/libspdlog.a: _deps/spdlog-build/CMakeFiles/spdlog.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/Projects/VirtualFluids_Develop/buildWSL/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Linking CXX static library libspdlog.a"
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && $(CMAKE_COMMAND) -P CMakeFiles/spdlog.dir/cmake_clean_target.cmake
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/spdlog.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
_deps/spdlog-build/CMakeFiles/spdlog.dir/build: _deps/spdlog-build/libspdlog.a
.PHONY : _deps/spdlog-build/CMakeFiles/spdlog.dir/build

_deps/spdlog-build/CMakeFiles/spdlog.dir/clean:
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build && $(CMAKE_COMMAND) -P CMakeFiles/spdlog.dir/cmake_clean.cmake
.PHONY : _deps/spdlog-build/CMakeFiles/spdlog.dir/clean

_deps/spdlog-build/CMakeFiles/spdlog.dir/depend:
	cd /mnt/d/Projects/VirtualFluids_Develop/buildWSL && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/Projects/VirtualFluids_Develop /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-src /mnt/d/Projects/VirtualFluids_Develop/buildWSL /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build /mnt/d/Projects/VirtualFluids_Develop/buildWSL/_deps/spdlog-build/CMakeFiles/spdlog.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : _deps/spdlog-build/CMakeFiles/spdlog.dir/depend

