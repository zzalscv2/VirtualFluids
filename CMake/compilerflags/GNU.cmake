
#############################################################################################################
# Flags
#############################################################################################################
LIST(APPEND CAB_COMPILER_ADDTIONAL_CXX_COMPILER_FLAGS "-O3 -fomit-frame-pointer -finline-functions -funroll-all-loops -fPIC")
LIST(APPEND CAB_COMPILER_ADDTIONAL_CXX_COMPILER_FLAGS "-Wno-deprecated") #deprecated header warning (jarl benutzt sstream weil schneller und so)

LIST(APPEND CAB_COMPILER_ADDTIONAL_C_COMPILER_FLAGS "-O3 -fomit-frame-pointer -finline-functions -funroll-all-loops -fPIC")

#############################################################################################################
# OpenMP support
#############################################################################################################
IF(USE_OPENMP)
    LIST(APPEND CAB_COMPILER_ADDTIONAL_CXX_COMPILER_FLAGS "-fopenmp")
    LIST(APPEND CAB_COMPILER_ADDTIONAL_C_COMPILER_FLAGS "-fopenmp")
ENDIF()

#############################################################################################################
# mt support
#############################################################################################################
LIST(APPEND CAB_COMPILER_ADDTIONAL_CXX_COMPILER_FLAGS "-pthread")
LIST(APPEND CAB_COMPILER_ADDTIONAL_C_COMPILER_FLAGS "-pthread")

#############################################################################################################
# c++ 11 support
#############################################################################################################
LIST(APPEND CAB_COMPILER_ADDTIONAL_CXX_COMPILER_FLAGS "-std=c++11")
LIST(APPEND CAB_COMPILER_ADDTIONAL_C_COMPILER_FLAGS "-std=c++11")

#############################################################################################################
# c++ 17 support
#############################################################################################################
#LIST(APPEND CAB_COMPILER_ADDTIONAL_CXX_COMPILER_FLAGS "-std=c++17")
#LIST(APPEND CAB_COMPILER_ADDTIONAL_C_COMPILER_FLAGS "-std=c++17")

#LIST(APPEND CAB_COMPILER_ADDTIONAL_CXX_COMPILER_FLAGS "-fext-numeric-literals")
#LIST(APPEND CAB_COMPILER_ADDTIONAL_CXX_COMPILER_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")
#LIST(APPEND CAB_COMPILER_ADDTIONAL_CXX_COMPILER_FLAGS "-Wregister")


IF(NOT APPLE)
    LIST(APPEND CAB_ADDITIONAL_LINK_PROPS "-lrt")
ENDIF()
