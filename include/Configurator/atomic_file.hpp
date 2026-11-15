/*
 * This file is part of the KADATH library.
 * Author: Samuel Tootle
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace Kadath {
namespace FUKA_Config {

/**
 * Solutions are written by rank 0 and read by every rank right after an MPI barrier, often under the same
 * name as the previous one (e.g. ns_regrid.dat for every step of a sequence). On a network file system a
 * rank on another node can then still read the old or a partly written file (seen as garbage sizes and
 * std::bad_alloc on the second node). Writing to a temporary file, syncing it, and renaming it into place
 * makes the new file appear complete and as a new inode.
 */
inline std::string atomic_tmp_name(std::string const& fname) {
    return fname + ".tmp." + std::to_string(::getpid());
}

/// Sync the written temporary file to storage and move it to its final name.
inline void atomic_commit(std::string const& tmp, std::string const& fname) {
    int fd = ::open(tmp.c_str(), O_RDONLY);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
    if (std::rename(tmp.c_str(), fname.c_str()) != 0) {
        std::remove(tmp.c_str());
        throw std::runtime_error("cannot rename " + tmp + " to " + fname);
    }
}

}  // namespace FUKA_Config
}  // namespace Kadath
