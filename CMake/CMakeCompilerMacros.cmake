###############################################################
# Aktivieren von ELSIF in CMake
###############################################################
SET(CMAKE_ALLOW_LOOSE_LOOP_CONSTRUCTS 1)

###############################################################
# IS_64BIT_SYSTEM (check put if OS is 64 bit compatible)
###############################################################
MACRO(IS_64BIT_SYSTEM is64BitOutVar)

  SET(${is64BitOutVar} FALSE)

  IF(APPLE)
       EXEC_PROGRAM( arch                            
                  ARGS -x86_64 echo "64bit"
                  OUTPUT_VARIABLE CAB_SYSTEM_INFO_1 )
    IF(${CAB_SYSTEM_INFO_1} MATCHES "64bit")
      SET(${is64BitOutVar} TRUE)
    ENDIF()

  ELSEIF(UNIX)

    EXEC_PROGRAM( uname                           
                  ARGS -m
                  OUTPUT_VARIABLE CAB_SYSTEM_INFO_1 )

    STRING(REGEX MATCH "x86_64"  CAB_SYSTEM_INFO_1  ${CAB_SYSTEM_INFO_1})
              

    EXEC_PROGRAM( getconf
                  ARGS -a | grep -i LONG_BIT
                  OUTPUT_VARIABLE CAB_SYSTEM_INFO_2 )

    STRING(REGEX MATCH "64"  CAB_SYSTEM_INFO_2  ${CAB_SYSTEM_INFO_2})

    IF(CAB_SYSTEM_INFO_1 STREQUAL  "x86_64" AND CAB_SYSTEM_INFO_2 STREQUAL "64")
      SET(${is64BitOutVar} TRUE)
    ENDIF()

  ELSEIF(WIN32)

    MESSAGE(STATUS "IS_64BIT_SYSTEM: determining system type (32/64bit)...(this may take a few moments)")
    EXEC_PROGRAM( SystemInfo OUTPUT_VARIABLE CAB_SYSTEM_INFO_1 )

    STRING(REGEX MATCH "x64-based PC"  CAB_SYSTEM_INFO_1  ${CAB_SYSTEM_INFO_1})

    IF(CAB_SYSTEM_INFO_1 MATCHES "x64-based PC")
      SET(${is64BitOutVar} TRUE)
      MESSAGE(STATUS "IS_64BIT_SYSTEM: determining system type (32/64bit)... done (-> 64 Bit)")
    ELSE()
      MESSAGE(STATUS "IS_64BIT_SYSTEM: determining system type (32/64bit)... done (-> 32 Bit)")
    ENDIF()

  ELSE()
    MESSAGE(FATAL_ERROR "IS_64BIT_SYSTEM: unknown OS")
  ENDIF()

ENDMACRO(IS_64BIT_SYSTEM is64BitOutVar)


###############################################################
### SET_CAB_COMPILER                                        ###
### Macro sets CAB_COMPILER variable if not set             ###
### for msvc:      CMake Variables are used                 ###
### for intel,gcc: --version call is evaluated              ###
###############################################################
MACRO(SET_CAB_COMPILER)
   IF(NOT CMAKE_CXX_COMPILER)
      MESSAGE(FATAL_ERROR "before SET_CAB_COMPILER-Macro PROJECT-Macro has to be called")
   ELSE()
      IF(NOT CAB_COMPILER)
         IF(MSVC)
		   IF(CMAKE_CL_64)
		     SET( CAB_COMPILER "msvc19_64" )
		   ELSE()
		     SET( CAB_COMPILER "msvc19_32" )
		   ENDIF()
         #ELSEIF(APPLE)
		 ELSEIF("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
            SET( CAB_COMPILER "clang" )
         ELSE()
           EXEC_PROGRAM( ${CMAKE_CXX_COMPILER}                          
                          ARGS --version 
                          OUTPUT_VARIABLE CAB_COMPILER_INFO )

            STRING(REGEX REPLACE ".* \\((.*)\\) ([0-9]*)\\.([0-9]*)[\\. ]([0-9]*).*" "\\1" CAB_COMPILER_NAME          ${CAB_COMPILER_INFO})
            STRING(REGEX REPLACE "[^ ]*[^0-9]*([0-9]*)\\.([0-9]*)[\\. ]([0-9]*)[^0-9]*.*" "\\1" CAB_COMPILER_VERSION_MAJOR ${CAB_COMPILER_INFO})
            STRING(REGEX REPLACE "[^ ]*[^0-9]*([0-9]*)\\.([0-9]*)[\\. ]([0-9]*)[^0-9]*.*" "\\2" CAB_COMPILER_VERSION_MINOR ${CAB_COMPILER_INFO})
            STRING(REGEX REPLACE "[^ ]*[^0-9]*([0-9]*)\\.([0-9]*)[\\. ]([0-9]*)[^0-9]*.*" "\\3" CAB_COMPILER_VERSION_PATCH ${CAB_COMPILER_INFO})

            STRING(TOLOWER ${CAB_COMPILER_NAME} CAB_COMPILER_NAME)

            IF(CMAKE_COMPILER_IS_GNUCXX)
               SET(CAB_COMPILER_NAME "gcc")
               SET(USE_GCC ON)
            ENDIF()
            
            SET(CAB_COMPILER "${CAB_COMPILER_NAME}${CAB_COMPILER_VERSION_MAJOR}${CAB_COMPILER_VERSION_MINOR}")
         ENDIF()
      ENDIF()

      SET(CAB_COMPILER ${CAB_COMPILER} CACHE STRING "compiler") 
   ENDIF()

ENDMACRO(SET_CAB_COMPILER)

################################################################
###               CHECK_FOR_VARIABLE                         ###
###  checks for a variable (also env-variables)
###  if not found -> error-message!!!
###  always: cache-entry update
################################################################
MACRO(CHECK_FOR_VARIABLE var)
   #check ob evtl enviromentvariable gesetzt
   IF(NOT ${var})  #true if ${var} NOT: empty, 0, N, NO, OFF, FALSE, NOTFOUND, or <variable>-NOTFOUND
     SET(${var} $ENV{${var}})
   ENDIF()

   IF(NOT DEFINED ${var})
      SET(${var} "${var}-NOTFOUND" CACHE STRING "${ARGN}" FORCE)
   ENDIF(NOT DEFINED ${var})

   IF(NOT ${var})
      MESSAGE(FATAL_ERROR "CHECK_FOR_VARIABLE - error - set ${var}")
   ENDIF()
   
SET(${var} ${${var}} CACHE STRING "${ARGN}" FORCE) 
ENDMACRO(CHECK_FOR_VARIABLE var)


#################################################################
###   ADD_CXX_FLAGS(flags)     				      ###
### flags will be added to CMAKE_CXX_FLAGS                    ###
#################################################################
MACRO(ADD_CXX_FLAGS)
  FOREACH(arg ${ARGN})
    SET(TMP ${arg}) #elsewise the Seperate command doesn't work)
    SEPARATE_ARGUMENTS(TMP)
    FOREACH(option ${TMP})
      STRING(REGEX REPLACE " ${option}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      STRING(REGEX REPLACE "${option}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${option}" CACHE STRING "common C++ build flags" FORCE)
    ENDFOREACH(option ${TMP})  
  ENDFOREACH(arg ${ARGN})
ENDMACRO(ADD_CXX_FLAGS)

#################################################################
###   ADD_CXX_FLAGS_IF(option flags)             	      ###
### flags will be added to CMAKE_CXX_FLAGS if option exists   ###
#################################################################
MACRO(ADD_CXX_FLAGS_IF condition)
  IF(${condition})
    ADD_CXX_FLAGS(${ARGN})
  ENDIF(${condition})
ENDMACRO(ADD_CXX_FLAGS_IF)

#################################################################
###   REMOVE_CXX_FLAGS(flags)     	  	              ###
### flags will be removed from CMAKE_CXX_FLAGS                ###
#################################################################
MACRO(REMOVE_CXX_FLAGS)
  FOREACH(arg ${ARGN})
    SET(TMP ${arg}) #elsewise the Seperate command doesn't work)
    SEPARATE_ARGUMENTS(TMP)
    FOREACH(option ${TMP})
      STRING(REGEX REPLACE " ${option}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
      STRING(REGEX REPLACE "${option}" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    ENDFOREACH(option ${TMP})  
  ENDFOREACH(arg ${ARGN})
ENDMACRO(REMOVE_CXX_FLAGS)

#####################################################################
###   REMOVE_CXX_FLAGS(option flags)     		          ###
### flags will be removed from CMAKE_CXX_FLAGS if option exists   ###
#####################################################################
MACRO(REMOVE_CXX_FLAGS_IF condition)
  IF(${condition})
    REMOVE_CXX_FLAGS(${ARGN})
  ENDIF(${condition})
ENDMACRO(REMOVE_CXX_FLAGS_IF)

#################################################################
###   ADD_CXX_BUILDTYPE_FLAGS(buildtype flags)   	      ###
### flags will be added to CMAKE_CXX_BUILDTYPE_FLAGS          ###
#################################################################
MACRO(ADD_CXX_BUILDTYPE_FLAGS buildtype)
  IF(CMAKE_CXX_FLAGS_${buildtype})
    FOREACH(arg ${ARGN})
      SET(TMP ${arg}) #elsewise the Seperate command doesn't work)
      SEPARATE_ARGUMENTS(TMP)
      FOREACH(option ${TMP})
        STRING(REGEX REPLACE " ${option}" "" CMAKE_CXX_FLAGS_${buildtype} "${CMAKE_CXX_FLAGS_${buildtype}}")
        STRING(REGEX REPLACE "${option}" "" CMAKE_CXX_FLAGS_${buildtype} "${CMAKE_CXX_FLAGS_${buildtype}}")
        SET(CMAKE_CXX_FLAGS_${buildtype} "${CMAKE_CXX_FLAGS_${buildtype}} ${option}" CACHE STRING "common C++ build flags for ${buildtype}" FORCE)
      ENDFOREACH(option ${TMP})  
    ENDFOREACH(arg ${ARGN})
  ENDIF(CMAKE_CXX_FLAGS_${buildtype})
ENDMACRO(ADD_CXX_BUILDTYPE_FLAGS)

#########################################################################
###   ADD_CXX_BUILDTYPE_FLAGS(buildtype option flags)	              ###
### flags will be added to CMAKE_CXX_BUILDTYPE_FLAGS if option exists ###
#########################################################################
MACRO(ADD_CXX_BUILDTYPE_FLAGS_IF buildtype condition)
  IF(${condition})
    ADD_CXX_BUILDTYPE_FLAGS(${buildtype} ${ARGN})
  ENDIF(${condition})
ENDMACRO(ADD_CXX_BUILDTYPE_FLAGS_IF)

#################################################################
###   REMOVE_CXX_BUILDTYPE_FLAGS(buildtype flags)             ###
### flags will be removed from CMAKE_CXX_FLAGS                ###
#################################################################
MACRO(REMOVE_CXX_BUILDTYPE_FLAGS buildtype)
  IF(CMAKE_CXX_FLAGS_${buildtype})
    FOREACH(arg ${ARGN})
      SET(TMP ${arg}) #elsewise the Seperate command doesn't work)
      SEPARATE_ARGUMENTS(TMP)
      FOREACH(option ${TMP})
	STRING(REGEX REPLACE " ${option}" "" CMAKE_CXX_FLAGS_${buildtype} "${CMAKE_CXX_FLAGS_${buildtype}}")
        STRING(REGEX REPLACE "${option}" "" CMAKE_CXX_FLAGS_${buildtype} "${CMAKE_CXX_FLAGS_${buildtype}}")
        SET(CMAKE_CXX_FLAGS_${buildtype} "${CMAKE_CXX_FLAGS_${buildtype}}" 
            CACHE STRING "C++ build flags for ${buildtype} configuration" FORCE)
      ENDFOREACH(option ${TMP})  
    ENDFOREACH(arg ${ARGN})
  ENDIF(CMAKE_CXX_FLAGS_${buildtype})
ENDMACRO(REMOVE_CXX_BUILDTYPE_FLAGS)

#####################################################################
###   REMOVE_CXX_BUILDTYPE_FLAGS_IF(buildtype option flags)       ###
### flags will be removed from CMAKE_CXX_FLAGS if option exists   ###
#####################################################################
MACRO(REMOVE_CXX_BUILDTYPE_FLAGS_IF condition)
  IF(${condition})
    REMOVE_CXX_BUILDTYPE_FLAGS(${buildtype} ${ARGN})
  ENDIF(${condition})
ENDMACRO(REMOVE_CXX_BUILDTYPE_FLAGS_IF)

#####################################################################
###   SET_CXX_COMPILER( compiler)                                 ###
### flags will be removed from CMAKE_CXX_FLAGS if option exists   ###
#####################################################################
#MACRO(SET_CXX_COMPILER compiler)
#  INCLUDE (CMakeForceCompiler)
#  SET(CMAKE_SYSTEM_NAME Generic)
#  CMAKE_FORCE_CXX_COMPILER   (${compiler} "set by user")
#  SET(CMAKE_CXX_COMPILER ${compiler} CACHE STRING "C++ compiler" FORCE)
#ENDMACRO(SET_CXX_COMPILER)

#################################################################
###   ADD_C_FLAGS(flags)     				      ###
### flags will be added to CMAKE_C_FLAGS                    ###
#################################################################
MACRO(ADD_C_FLAGS)
  FOREACH(arg ${ARGN})
    SET(TMP ${arg}) #elsewise the Seperate command doesn't work)
    SEPARATE_ARGUMENTS(TMP)
    FOREACH(option ${TMP})
      STRING(REGEX REPLACE " ${option}" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
      STRING(REGEX REPLACE "${option}" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
      SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${option}" CACHE STRING "common C++ build flags" FORCE)
    ENDFOREACH(option ${TMP})  
  ENDFOREACH(arg ${ARGN})
ENDMACRO(ADD_C_FLAGS)

#################################################################
###   ADD_C_FLAGS(option flags)     			      ###
### flags will be added to CMAKE_C_FLAGS if option exists     ###
#################################################################
MACRO(ADD_C_FLAGS_IF condition)
  IF(${condition})
    ADD_C_FLAGS(${ARGN})
  ENDIF(${condition})
ENDMACRO(ADD_C_FLAGS_IF)

#################################################################
###   REMOVE_C_FLAGS(flags)     	  	              ###
### flags will be removed from CMAKE_C_FLAGS                  ###
#################################################################
MACRO(REMOVE_C_FLAGS)
  FOREACH(arg ${ARGN})
    SET(TMP ${arg}) #elsewise the Seperate command doesn't work)
    SEPARATE_ARGUMENTS(TMP)
    FOREACH(option ${TMP})
      STRING(REGEX REPLACE " ${option}" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
      STRING(REGEX REPLACE "${option}" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    ENDFOREACH(option ${TMP})  
  ENDFOREACH(arg ${ARGN})
ENDMACRO(REMOVE_C_FLAGS)

#####################################################################
###   REMOVE_C_FLAGS(option flags)                                ###
### flags will be removed from CMAKE_C_FLAGS if option exists     ###
#####################################################################
MACRO(REMOVE_C_FLAGS_IF condition)
  IF(${condition})
    REMOVE_C_FLAGS(${ARGN})
  ENDIF(${condition})
ENDMACRO(REMOVE_C_FLAGS_IF)

#################################################################
###   ADD_C_BUILDTYPE_FLAGS(buildtype flags)                  ###
### flags will be added to CMAKE_C_BUILDTYPE_FLAGS            ###
#################################################################
MACRO(ADD_C_BUILDTYPE_FLAGS buildtype)
  IF(CMAKE_C_FLAGS_${buildtype})
    FOREACH(arg ${ARGN})
      SET(TMP ${arg}) #elsewise the Seperate command doesn't work)
      SEPARATE_ARGUMENTS(TMP)
      FOREACH(option ${TMP})
        STRING(REGEX REPLACE " ${option}" "" CMAKE_C_FLAGS_${buildtype} "${CMAKE_C_FLAGS_${buildtype}}")
        STRING(REGEX REPLACE "${option}" "" CMAKE_C_FLAGS_${buildtype} "${CMAKE_C_FLAGS_${buildtype}}")
        SET(CMAKE_C_FLAGS_${buildtype} "${CMAKE_C_FLAGS_${buildtype}} ${option}" CACHE STRING "common C++ build flags for ${buildtype}" FORCE)
      ENDFOREACH(option ${TMP})  
    ENDFOREACH(arg ${ARGN})
  ENDIF(CMAKE_C_FLAGS_${buildtype})
ENDMACRO(ADD_C_BUILDTYPE_FLAGS)

#########################################################################
###   ADD_C_BUILDTYPE_FLAGS(buildtype option flags)	              ###
### flags will be added to CMAKE_C_BUILDTYPE_FLAGS if option exists ###
#########################################################################
MACRO(ADD_C_BUILDTYPE_FLAGS_IF buildtype condition)
  IF(${condition})
    ADD_C_BUILDTYPE_FLAGS(${buildtype} ${ARGN})
  ENDIF(${condition})
ENDMACRO(ADD_C_BUILDTYPE_FLAGS_IF)

#################################################################
###   REMOVE_C_BUILDTYPE_FLAGS(buildtype flags)               ###
### flags will be removed from CMAKE_C_FLAGS                  ###
#################################################################
MACRO(REMOVE_C_BUILDTYPE_FLAGS buildtype)
  IF(CMAKE_C_FLAGS_${buildtype})
    FOREACH(arg ${ARGN})
      SET(TMP ${arg}) #elsewise the Seperate command doesn't work)
      SEPARATE_ARGUMENTS(TMP)
      FOREACH(option ${TMP})
	STRING(REGEX REPLACE " ${option}" "" CMAKE_C_FLAGS_${buildtype} "${CMAKE_C_FLAGS_${buildtype}}")
        STRING(REGEX REPLACE "${option}" "" CMAKE_C_FLAGS_${buildtype} "${CMAKE_C_FLAGS_${buildtype}}")
        SET(CMAKE_C_FLAGS_${buildtype} "${CMAKE_C_FLAGS_${buildtype}}" 
            CACHE STRING "C++ build flags for ${buildtype} configuration" FORCE)
      ENDFOREACH(option ${TMP})  
    ENDFOREACH(arg ${ARGN})
  ENDIF(CMAKE_C_FLAGS_${buildtype})
ENDMACRO(REMOVE_C_BUILDTYPE_FLAGS)

#####################################################################
###   REMOVE_C_BUILDTYPE_FLAGS_IF(buildtype option flags)         ###
### flags will be removed from CMAKE_C_FLAGS if option exists     ###
#####################################################################
MACRO(REMOVE_C_BUILDTYPE_FLAGS_IF condition)
  IF(${condition})
    REMOVE_C_BUILDTYPE_FLAGS(${buildtype} ${ARGN})
  ENDIF(${condition})
ENDMACRO(REMOVE_C_BUILDTYPE_FLAGS_IF)

#####################################################################
###   SET_C_COMPILER( compiler)                                   ###
### flags will be removed from CMAKE_C_FLAGS if option exists     ###
#####################################################################
MACRO(SET_C_COMPILER compiler)
  INCLUDE (CMakeForceCompiler)
  SET(CMAKE_SYSTEM_NAME Generic)
  CMAKE_FORCE_C_COMPILER  (${compiler} "set by user")
  SET(CMAKE_C_COMPILER ${compiler} CACHE STRING "C compiler" FORCE)
ENDMACRO(SET_C_COMPILER)

#################################################################
###   ADD_EXE_LINKER_FLAGS(flags)                             ###
### flags will be added to CMAKE_EXE_LINKER_FLAGS             ###
#################################################################
MACRO(ADD_EXE_LINKER_FLAGS)
  FOREACH(arg ${ARGN})
    SET(TMP ${arg}) #elsewise the Seperate command doesn't work)
    SEPARATE_ARGUMENTS(TMP)
    FOREACH(option ${TMP})
      STRING(REGEX REPLACE " ${option}" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
      STRING(REGEX REPLACE "${option}" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
      SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${option}" CACHE STRING "common C++ build flags" FORCE)
    ENDFOREACH(option ${TMP})  
  ENDFOREACH(arg ${ARGN})
ENDMACRO(ADD_EXE_LINKER_FLAGS)
