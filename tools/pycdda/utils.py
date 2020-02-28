# -*- coding: utf-8 -*-

"""
The MIT License (MIT)

Copyright (c) 2020 Imayhaveborkedit

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

import shutil
import typing
import textwrap

prim_types = (str, int, float, bool)
container_types = (dict, list)

json_prim = typing.Union[str, int, float, bool, None]
json_type = typing.Union[dict, list, json_prim]

int64_max = 2 ** 63 - 1
int64_min = -2 ** 63


def wrap(text:str, x:int=None, *, indent=0, **kwargs) -> str:
    if x is None or x <= 0:
        x = shutil.get_terminal_size().columns - 2

    kwargs.setdefault('subsequent_indent', ' '*indent)
    _kwargs = {
        'break_on_hyphens': False,
        **kwargs
    }

    return '\n'.join(textwrap.wrap(text, x, **kwargs))
