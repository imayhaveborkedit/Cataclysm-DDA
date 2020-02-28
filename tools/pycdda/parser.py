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

import sys
import json

from .utils import json_type

__all__ = ['parse_json', 'JSONDecodeError']


class JSONDecodeError(Exception):
    def __init__(self, msg, formatted_msg, original=None):
        super().__init__(msg)
        self.formatted_msg = formatted_msg
        self.original = original

    def print_error(self, *, file=sys.stderr, **kwargs):
        """Prints a nicely formatted message indicating the error."""
        print(self.formatted_msg, file=file, **kwargs)


def parse_json(data:str, source='<string>', *, unique_key_hook=None) -> json_type:
    """Like json.loads, but with a fancy exception for parsing errors."""

    try:
        return json.loads(data, object_pairs_hook=unique_key_hook)
    except json.JSONDecodeError as e:
        json_lines = data.split('\n')

        location = source if source[0] == '<' else f'"{source}"'

        location_header = f"File {location}, line {e.lineno}, column {e.colno} (char {e.pos})"
        arrow_header    =  "---> {0: >{1}} "
        empty_header    =  "     {0: >{1}} "
        pointer_header  = f"{'^': >{e.colno}}"
        error_header    = f"Syntax error: {e.msg}"

        # Pick the lines of the input data to display
        start_line = max(0, e.lineno-3) + 1
        displayed_lines = json_lines[start_line-1:e.lineno]

        # TODO: shift lines over to the longest line or wrap pointer
        #       in the case of a long line, truncate lines with `: ` and `:+`
        #       or if the target line is a long line, shift them over and prepend `:`

        # lengthen the pointer line
        longest_num_len = len(str(e.lineno))
        arrow_header_len = len(arrow_header.format(e.lineno, longest_num_len))
        complete_pointer_header = ' ' * arrow_header_len + pointer_header

        completed_displayed_lines = []

        # Combine line numbers and data lines
        for line_no, line in enumerate(displayed_lines, start_line):
            header = arrow_header if line_no == e.lineno else empty_header
            completed_displayed_lines.append(header.format(line_no, longest_num_len)+line)

        complete_error_message = '\n'.join((location_header,
                                            *completed_displayed_lines,
                                            complete_pointer_header,
                                            error_header))

        raise parsejson.JSONDecodeError(str(e), complete_error_message, e) from None
