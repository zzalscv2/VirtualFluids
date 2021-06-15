#################################################################################
# VirtualFluids MACHINE FILE
# Responsible: Martin Schoenherr
# OS:          Windows 10
#################################################################################

#Don't change:
SET(METIS_ROOT ${CMAKE_SOURCE_DIR}/3rdParty/metis/metis-5.1.0 CACHE PATH "METIS ROOT")
SET(GMOCK_ROOT ${CMAKE_SOURCE_DIR}/3rdParty/googletest CACHE PATH "GMOCK ROOT")
SET(JSONCPP_ROOT ${CMAKE_SOURCE_DIR}/3rdParty/jsoncpp CACHE PATH "JSONCPP ROOT")
SET(FFTW_ROOT ${CMAKE_SOURCE_DIR}/3rdParty/fftw/fftw-3.3.7 CACHE PATH "JSONCPP ROOT")

SET(CMAKE_CUDA_ARCHITECTURES 52)

SET(VTK_DIR "F:/Libraries/vtk/VTK-8.2.0/build" CACHE PATH "VTK directory override" FORCE)

SET(PATH_NUMERICAL_TESTS "E:/temp/numericalTests/")
LIST(APPEND VF_COMPILER_DEFINITION "PATH_NUMERICAL_TESTS=${PATH_NUMERICAL_TESTS}")
