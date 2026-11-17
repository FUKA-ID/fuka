cmake_minimum_required(VERSION 3.12)
project(fuka_solvers_tests)

include ($ENV{HOME_KADATH}/Cmake/CMakeExec.cmake)

foreach(PROG test_solution_snapshot)
  add_executable(${PROG} src/${PROG}.cpp ${HEADERS})
  target_link_libraries(${PROG} ${LIB_KADATH} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES} ${BLACS_LIBRARIES} ${GRHAYL_LIBRARIES})
endforeach()
