r"""
=======================================================================================
 ____          ____    __    ______     __________   __      __       __        __
 \    \       |    |  |  |  |   _   \  |___    ___| |  |    |  |     /  \      |  |
  \    \      |    |  |  |  |  |_)   |     |  |     |  |    |  |    /    \     |  |
   \    \     |    |  |  |  |   _   /      |  |     |  |    |  |   /  /\  \    |  |
    \    \    |    |  |  |  |  | \  \      |  |     |   \__/   |  /  ____  \   |  |____
     \    \   |    |  |__|  |__|  \__\     |__|      \________/  /__/    \__\  |_______|
      \    \  |    |   ________________________________________________________________
       \    \ |    |  |  ______________________________________________________________|
        \    \|    |  |  |         __          __     __     __     ______      _______
         \         |  |  |_____   |  |        |  |   |  |   |  |   |   _  \    /  _____)
          \        |  |   _____|  |  |        |  |   |  |   |  |   |  | \  \   \_______
           \       |  |  |        |  |_____   |   \_/   |   |  |   |  |_/  /    _____  |
            \ _____|  |__|        |________|   \_______/    |__|   |______/    (_______/

  This file is part of VirtualFluids. VirtualFluids is free software: you can
  redistribute it and/or modify it under the terms of the GNU General Public
  License as published by the Free Software Foundation, either version 3 of
  the License, or (at your option) any later version.

  VirtualFluids is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
  for more details.

  You should have received a copy of the GNU General Public License along
  with VirtualFluids (see COPYING.txt). If not, see <http://www.gnu.org/licenses/>.

! \file logger.pyi
! \ingroup basics
! \author Henry Korb
=======================================================================================
"""
from typing import Any, ClassVar

log: None

class Level:
    __members__: ClassVar[dict] = ...  # read-only
    INFO_HIGH: ClassVar[Level] = ...
    INFO_INTERMEDIATE: ClassVar[Level] = ...
    INFO_LOW: ClassVar[Level] = ...
    LOGGER_ERROR: ClassVar[Level] = ...
    WARNING: ClassVar[Level] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...

class Logger:
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def add_stdout() -> None: ...
    @staticmethod
    def enable_printed_rank_numbers(print: bool) -> None: ...
    @staticmethod
    def set_debug_level(level: int) -> None: ...
    @staticmethod
    def time_stamp(time_stemp: TimeStamp) -> None: ...

class TimeStamp:
    __members__: ClassVar[dict] = ...  # read-only
    DISABLE: ClassVar[TimeStamp] = ...
    ENABLE: ClassVar[TimeStamp] = ...
    __entries: ClassVar[dict] = ...
    def __init__(self, arg0: int) -> None: ...
    def __eq__(self, arg0: object) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __int__(self) -> int: ...
    def __ne__(self, arg0: object) -> bool: ...
    def __setstate__(self, arg0: int) -> None: ...
    @property
    def name(self) -> str: ...