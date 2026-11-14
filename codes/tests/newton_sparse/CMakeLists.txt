cmake_minimum_required(VERSION 3.12)
project(newton_sparse_tests)

include ($ENV{HOME_KADATH}/Cmake/CMakeExec.cmake)

foreach(PROG test_sparse_triplets test_mumps_wrapper test_newton_kadath replay_jacobian)
  add_executable(${PROG} src/${PROG}.cpp ${HEADERS})
  target_link_libraries(${PROG} ${LIB_KADATH} ${GSL_LIBRARIES} ${FFTW_LIBRARIES} ${LAPACK_LIBRARIES} ${SCALAPACK_LIBRARIES} ${BLACS_LIBRARIES})
endforeach()
