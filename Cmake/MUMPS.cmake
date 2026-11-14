# Optional MUMPS sparse direct solver for the parallel Newton-Raphson step.
#
# Enabled when PAR_VERSION is ON, the MUMPS option is ON (default) and the
# MUMPS_ROOT environment variable points at a MUMPS installation (dmumps_c.h,
# libdmumps). METIS_ROOT / PARMETIS_ROOT are added when set.
#
# Sets: KADATH_USE_MUMPS (bool), MUMPS_INCLUDE_DIRS, MUMPS_LIBRARIES and adds
# -DKADATH_USE_MUMPS. The libraries are prepended to SCALAPACK_LIBRARIES so that
# every executable linking ${SCALAPACK_LIBRARIES} picks them up without edits.

option(MUMPS "Use the MUMPS sparse solver for the Newton step" ON)
set(KADATH_USE_MUMPS OFF)

if(PAR_VERSION AND MUMPS)
  if("$ENV{MUMPS_ROOT}" STREQUAL "")
    message(STATUS "MUMPS: MUMPS_ROOT not set, building with the dense ScaLAPACK Newton solver only")
  else()
    find_path(MUMPS_INCLUDE_DIR dmumps_c.h PATHS $ENV{MUMPS_ROOT}/include NO_DEFAULT_PATH)
    find_library(DMUMPS_LIBRARY dmumps PATHS $ENV{MUMPS_ROOT}/lib $ENV{MUMPS_ROOT}/lib64 NO_DEFAULT_PATH)
    find_library(MUMPS_COMMON_LIBRARY mumps_common PATHS $ENV{MUMPS_ROOT}/lib $ENV{MUMPS_ROOT}/lib64 NO_DEFAULT_PATH)
    find_library(MUMPS_PORD_LIBRARY pord PATHS $ENV{MUMPS_ROOT}/lib $ENV{MUMPS_ROOT}/lib64 NO_DEFAULT_PATH)

    if(NOT MUMPS_INCLUDE_DIR OR NOT DMUMPS_LIBRARY OR NOT MUMPS_COMMON_LIBRARY)
      message(FATAL_ERROR "MUMPS: MUMPS_ROOT=$ENV{MUMPS_ROOT} does not contain dmumps_c.h / libdmumps / libmumps_common")
    endif()

    set(MUMPS_INCLUDE_DIRS ${MUMPS_INCLUDE_DIR})
    set(MUMPS_LIBRARIES ${DMUMPS_LIBRARY} ${MUMPS_COMMON_LIBRARY})
    if(MUMPS_PORD_LIBRARY)
      list(APPEND MUMPS_LIBRARIES ${MUMPS_PORD_LIBRARY})
    endif()

    foreach(ordering PARMETIS METIS)
      string(TOLOWER ${ordering} ordering_lib)
      if(NOT "$ENV{${ordering}_ROOT}" STREQUAL "")
        find_library(${ordering}_LIBRARY ${ordering_lib} PATHS $ENV{${ordering}_ROOT}/lib $ENV{${ordering}_ROOT}/lib64 NO_DEFAULT_PATH)
        if(${ordering}_LIBRARY)
          list(APPEND MUMPS_LIBRARIES ${${ordering}_LIBRARY})
        endif()
      endif()
    endforeach()
    list(APPEND MUMPS_LIBRARIES gfortran)

    # A shared libdmumps built against threaded MKL pulls in libmkl_gnu_thread;
    # MKL forbids two threading layers in one process, so match it.
    if(SCALAPACK_LIBRARIES MATCHES "mkl_sequential")
      string(REPLACE "-lmkl_sequential" "-lmkl_gnu_thread" SCALAPACK_LIBRARIES "${SCALAPACK_LIBRARIES}")
      message(STATUS "MUMPS: switched MKL threading layer to mkl_gnu_thread (run with OMP_NUM_THREADS=1)")
    endif()

    set(SCALAPACK_LIBRARIES "${MUMPS_LIBRARIES};${SCALAPACK_LIBRARIES}")
    include_directories(${MUMPS_INCLUDE_DIRS})
    add_definitions(-DKADATH_USE_MUMPS)
    set(KADATH_USE_MUMPS ON)
    message(STATUS "MUMPS: enabled (${MUMPS_LIBRARIES})")
  endif()
endif()
