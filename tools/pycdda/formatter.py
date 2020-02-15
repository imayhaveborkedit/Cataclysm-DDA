#!/usr/bin/env python3

"""
Port of json linter (tools/format/format.cpp) in python
Original linter: http://dev.narc.ro/cataclysm/format.html
with source at: https://github.com/CleverRaven/Cataclysm-DDA/blob/master/tools/format/format.cpp
"""

import io
import os
import re
import sys
import glob
import json
import time
import shlex
import shutil
import typing
import difflib
import pathlib
import argparse
import textwrap
import contextlib

from datetime import datetime
from ctypes import c_longlong


__all__ = ['JSONFormatter', 'parse_json', 'JSONDecodeError']


prim_types = (str, int, float, bool)
container_types = (dict, list)

json_prim = typing.Union[str, int, float, bool, None]
json_type = typing.Union[dict, list, json_prim]


class JSONDecodeError(Exception):
    def __init__(self, msg, formatted_msg, original=None):
        super().__init__(msg)
        self.formatted_msg = formatted_msg
        self.original = original

    def print_error(self, *, file=sys.stderr, **kwargs):
        """Prints a nicely formatted message indicating the error."""
        print(self.formatted_msg, file=file, **kwargs)

# These next three things are used for ensuring multiple comment keys in
# the same json object persist after being loaded.  Since python dicts
# require unique keys, we decode them as a special bytes object which we
# can later identify and encode back to "//".  This is done since there
# is no easy way to override the json encoder for basic types (str).

class _commentstr(bytes):
    def __new__(cls, *args, **kwargs):
        encoding = kwargs.pop('encoding', 'utf8')
        return super().__new__(cls, *args, encoding=encoding, **kwargs)

    # Needed when overriding __eq__
    def __hash__(self):
        return super().__hash__()

    def __len__(self):
        return 2

    def __eq__(self, other):
        return other == '//'

def _commentstr_hook(pairs):
    com_count = 0
    for i, (k, v) in enumerate(pairs):
        if k == '//':
            pairs[i] = (_commentstr(f'{k}{com_count}'), v)
            com_count += 1

    return dict(pairs)

class _SpecialEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, _commentstr):
            return '//'
        return super().default(o)

_json_dump_args = {
    'ensure_ascii': False,
    'cls': _SpecialEncoder
}

_isbool = lambda s: s.lower() in ('y', 'yes', 'true', '1')

_formatting_choices = {
    'indent-width': int,
    'max-line-length': int,
    'wrap-overrides': lambda s: s.split(','),
    'no-wrap-overrides': _isbool,
    'auto-split-stringarray-lines': _isbool,
    'always-wrap-depth': int,
    'cast-ints': _isbool,
    'round-floats': _isbool,
}

_warning_choices = ['all',
                    'stringarray-spaces',
                    'long-list-item',
                    'int-wraparound',
                    'float-rounding']

_default_output_template = '{path}'
_default_backup_template = '{path}_backup{suffix}'

def _static_template_vars() -> dict:
    return {
        'unixtime': time.time(),
        'datetime': datetime.now(),
        'utcdatetime': datetime.utcnow()
    }

def _dynamic_template_vars(file:str) -> dict:
    path = pathlib.Path(file).expanduser().resolve()
    return {
        'filename': path.name,
        'stem': path.stem,
        'ext': path.suffix.lstrip('.'),
        'suffix': path.suffix,
        'path': file,
        'absolute_path': path.absolute(),
        'parent': path.parent,
        'sep': os.path.sep,
        'cwd': path.cwd(),
        'home': path.home(),
        'stat': path.stat(),
        'input_pos': -1, # via kwargs
        'input_total': -1 # via kwargs
    }

def _render_template(template:str, file:str,*, variables:dict=None) -> str:
    if not variables:
        variables = {**_dynamic_template_vars(file), **_static_template_vars()}
    return template.format(**variables)


class JSONFormatter:
    """JSON Formatter

    Parameters
    -----------
    indent_width: Optional[int]
    max_line_length: Optional[int]
    wrap_overrides: Optional[List[str]]
    no_wrap_overrides: Optional[bool]
    autosplit: Optional[bool]
    cast_ints: Optional[bool]
    round_floats: Optional[bool]
    debug: Optional[bool]
    """

    default_wrap_overrides = ('rows', 'blueprint')

    def __init__(self, **kwargs):
        self.indent_width      = kwargs.pop('indent_width', 2)
        self.max_line_length   = kwargs.pop('max_line_length', 120)
        self.wrap_overrides    = kwargs.pop('wrap_overrides', self.default_wrap_overrides)
        self.no_wrap_overrides = kwargs.pop('no_wrap_overrides', False)
        self.autosplit         = kwargs.pop('autosplit', False)
        self.always_wrap_depth = kwargs.pop('always_wrap_depth', 1)
        self.cast_ints         = kwargs.pop('cast_ints', True)
        self.round_floats      = kwargs.pop('round_floats', True)
        self.debug             = kwargs.pop('debug', False)

        self._depth = 0
        self._buffer = io.StringIO()

    def _dbug(self, *args, file=sys.stderr, **kwargs):
        if self.debug:
            print(*args, file=file, **kwargs)

    def _reset(self):
        self._depth = 0
        self._buffer = io.StringIO()

    def _write_to_buffer(self, text: str):
        self._buffer.write(text)

    @property
    def current_indent(self) -> str:
        """Returns indentation for the current depth."""
        return ' ' * self.indent_width * self._depth

    def _get_container_width(self, obj: container_types) -> int:
        """Returns the width of a formatted json object or array from a dict or list."""

        if not obj:
            return 4 # empty containers are always 2 brackets with 2 spaces inside

        total = 2 # start with brackets
        items = obj if isinstance(obj, list) else obj.items()

        for item in items:
            # Recursively check size of sub-containers
            if isinstance(item, container_types):
                total += self._get_container_width(item) + 2 # comma + space

            # Check size of object pairs
            elif isinstance(item, tuple):
                k, v = item
                total += len(json.dumps(k, **_json_dump_args)) + 2 # keylength + colon + space

                # Recursively check size of sub-container values
                if isinstance(v, container_types):
                    total += self._get_container_width(v) + 2 # comma + space
                else:
                    total += len(json.dumps(v, **_json_dump_args)) + 2 # comma + space

            else:
                total += len(json.dumps(item, **_json_dump_args)) + 2 # comma + space

        # the extra two chars from the last space and comma count as the bracket padding
        return total

    def _should_wrap_container(self, obj: container_types) -> bool:
        """Returns True if a formatted container will need to be wrapped instead of inlined."""

        line_length = self._get_container_width(obj)
        will_wrap = line_length > self.max_line_length or self._depth <= self.always_wrap_depth
        will_wrap = will_wrap or self._has_wrap_override_key(obj)

        return will_wrap

    def _has_wrap_override_key(self, obj: container_types) -> bool:
        """Determine if a container has a container that will wrap due to an override"""

        if self.no_wrap_overrides:
            return False

        if isinstance(obj, list):
            for item in obj:
                if self._has_wrap_override_key(item):
                    return True

        elif isinstance(obj, dict):
            for key, value in obj.items():
                if key in self.wrap_overrides and isinstance(value, container_types) and len(value) > 1:
                    return True
                elif self._has_wrap_override_key(value):
                    return True

        return False

    @contextlib.contextmanager
    def _write_object(self, *, needs_indent: bool, wrap=False):
        """
        Context manager for setting up formatting for an object. Writes the
        opening { bracket upon entering, increases indent depth, writes the
        closing } bracket upon exit, and restores the indent depth. The
        caller is responsible for all the formatting inbetween.
        """

        if needs_indent:
            self._write_to_buffer(self.current_indent)

        self._write_to_buffer("{")
        self._depth += 1

        try:
            yield
        finally:
            self._depth -= 1
            if wrap:
                self._write_to_buffer(self.current_indent)
            self._write_to_buffer("}")

    @contextlib.contextmanager
    def _write_array(self, *, needs_indent: bool, wrap=False):
        """
        Context manager for setting up formatting for an array. Writes the
        opening [ bracket upon entering, increases indent depth, writes the
        closing ] bracket upon exit, and restores the indent depth. The
        caller is responsible for all the formatting inbetween.
        """

        if needs_indent:
            self._write_to_buffer(self.current_indent)

        self._write_to_buffer("[")
        self._depth += 1

        try:
            yield
        finally:
            self._depth -= 1
            if wrap:
                self._write_to_buffer(self.current_indent)
            self._write_to_buffer("]")

    def _write_object_kv(self, key: str, value: json_type, *, wrap: bool, is_last=False):
        """Writes an object kv pair, formatting the value."""

        # praise short circuiting
        force_wrap = not self.no_wrap_overrides and \
                     key in self.wrap_overrides and \
                     isinstance(value, container_types) and \
                     len(value) > 1

        wrap = wrap or force_wrap

        if wrap: # We're not inlined so we need an indent
            self._write_to_buffer(self.current_indent)

        # ensure that keys are written as strings -----------------vv
        self._write_to_buffer(f'{json.dumps(key, **_json_dump_args)!s}: ')
        self._write(value, needs_indent=False, is_last=is_last, force_wrap=force_wrap)

        if not is_last:
            self._write_to_buffer(',')

        self._write_to_buffer('\n' if wrap else ' ')

    def _write(self, obj: json_type, *, needs_indent=False, is_last=False, force_wrap=False):
        """Main function for recursively formatting and writing json items."""

        # basic types that don't need any special formatting
        if isinstance(obj, (*prim_types, _commentstr)) or obj is None:
            if needs_indent:
                self._write_to_buffer(self.current_indent)

            if self.cast_ints and isinstance(obj, int) and not isinstance(obj, bool):
                obj = c_longlong(obj).value

            elif self.round_floats and isinstance(obj, float):
                obj = round(obj, 6)

            self._write_to_buffer(json.dumps(obj, **_json_dump_args))

        # objects
        elif isinstance(obj, dict):
            wrap = self._should_wrap_container(obj) or force_wrap

            with self._write_object(needs_indent=needs_indent, wrap=wrap):
                if not obj:
                    # we're already inside the current object, so depth is one higher
                    if self._depth == 2:
                        # write special cased forced indentation
                        self._write_to_buffer(f'\n{self.current_indent}\n')
                    else:
                        self._write_to_buffer('  ')
                    return

                self._write_to_buffer('\n' if wrap else ' ')

                objlen = len(obj)

                for i, (k, v) in enumerate(obj.items(), 1):
                    self._write_object_kv(k, v, wrap=wrap, is_last=i == objlen)

        # arrays
        elif isinstance(obj, list):
            wrap = self._should_wrap_container(obj) or force_wrap

            with self._write_array(needs_indent=needs_indent, wrap=wrap):
                if not obj:
                    # We're already inside the current array, so its 2
                    if self._depth == 2:
                        # Write special cased indentation
                        self._write_to_buffer(f'\n{self.current_indent}\n')
                    else:
                        self._write_to_buffer('  ')
                    return

                self._write_to_buffer('\n' if wrap else ' ')

                if self.autosplit and len(obj) == 1 and isinstance(obj[0], str):
                    _text = obj.pop()
                    obj.extend(textwrap.wrap(text, self.max_line_length, break_on_hyphens=False))

                objlen = len(obj)

                for i, item in enumerate(obj, 1):
                    self._write(item, needs_indent=wrap, is_last=i == objlen)

                    # I hate 4-way conditionals
                    if i == objlen:
                        self._write_to_buffer('\n' if wrap else ' ')
                    else:
                        self._write_to_buffer(',\n' if wrap else ', ')


    def format_json(self, data: json_type) -> str:
        """Main entry point. Returns a formatted json string from a json-compatible python object."""

        self._reset()
        self._write(data)
        self._write_to_buffer('\n')
        return self._buffer.getvalue()


def parse_json(data:str, source='<string>', *, unique_key_hook=True) -> json_type:
    """Like json.loads, but with a fancy exception for parsing errors."""

    try:
        return json.loads(data, object_pairs_hook=_commentstr_hook if unique_key_hook else None)
    except json.JSONDecodeError as e:
        json_lines = data.split('\n')

        location = source if source[0] == '<' else f'"{source}"'

        location_header = f"File {location}, line {e.lineno}, column {e.colno} (char {e.pos})"
        arrow_header    =  "---> {0: >{1}} "
        empty_header    =  "     {0: >{1}} "
        pointer_header  = f"{'^': >{e.colno}}"
        error_header    = f"Syntax error: {e.msg}"

        # I wasted so much time on this one function and I don't even need it ;_;
        # def choose_lines(lineno, lines):
        #     return max(0, lineno-3), min(lineno+2, len(lines)), lines[max(0, lineno-3):min(lineno+2, len(lines))]

        # Pick the lines of the input data to display
        start_line = max(0, e.lineno-3) + 1
        displayed_lines = json_lines[start_line-1:e.lineno]

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

        raise JSONDecodeError(str(e), complete_error_message, e) from None

def main(io_mapping: dict, *,
         output_template: typing.Optional[str],
         backup_template: typing.Optional[str],
         backup: bool,
         overwrite: bool,
         verbose: bool,
         quiet: bool,
         continue_on_err: bool,
         parse: bool,
         diff: bool,
         formatter: list,
         warnings: list,
         **kwargs):

    STDIN = STDOUT = '-'

    # Extra flag consistancy logic, can change to raise errors instead
    if parse:
        diff = False
        formatter = []
        warnings = [] # TODO: allow parse with warnings

    if quiet:
        verbose = False

    if len(io_mapping) == 1 and tuple(io_mapping)[0] == STDIN:
        backup = False

    # Define some helper functions and variables
    _noop = lambda *a, **kw: None

    def print_err(*a, **kw):
        kw.setdefault('file', sys.stderr)
        print(*a, **kw)

    print_mesg = _noop if quiet else print
    print_more = print if verbose else _noop # TODO: add more print_more's
    maybe_exit = _noop if continue_on_err else exit

    progress = {
        'pass': 0,
        'fail': 0,
        'error': 0,
        'exists': 0,
        'remaining': len(io_mapping),
        'total': len(io_mapping)
    }

    maybe_exception = Exception if continue_on_err else ()

    # Early check for existing files
    if not overwrite and not output_template and not continue_on_err:
        for input_file, config in io_mapping.items():
            dest = config['out']

            if dest == '-' or dest is None:
                continue

            if os.path.exists(dest):
                print_err(FileExistsError(f"File {dest!r} (for input {input_file!r}) already exists"))
                exit(2)

    # translate formatter kv inputs to kwargs
    formatter_kwargs = {}
    for kv in formatter:
        k, v = kv[0].split('=')
        formatter_kwargs[k.replace('-','_')] = _formatting_choices[k](v)

    static_vars = _static_template_vars()

    # main run loop
    for input_file, config in io_mapping.items():
        if input_file == STDIN:
            source = '<stdin>'
            data = sys.stdin.read()
        else:
            source = input_file
            with open(input_file, encoding='utf8') as f:
                data = f.read()

        # set up try-finally block to always increment the progress counter
        try:
            # Load json and parse for syntax errors
            try:
                json_data = parse_json(data, source)
            except JSONDecodeError as err:
                progress['error'] += 1
                err.print_error()
                maybe_exit(2)
                continue

            # don't need to do anything else if we're only parsing
            if parse:
                progress['pass'] += 1
                continue

            # TODO: impl warnings
            # actually run the formatter
            result = formatted_json = JSONFormatter(**formatter_kwargs).format_json(json_data)

            # generate the format template variables
            _tmpl_vars = {
                **_dynamic_template_vars(input_file),
                **static_vars,
                'input_pos': progress['total'] - progress['remaining'] + 1,
                'input_total': progress['total']
            }

            # set/render output filenames and backup names
            output_filename = config['out']

            if output_filename is None:
                output_filename = os.path.devnull

            elif output_filename != STDOUT:
                output_filename = _render_template(config['out_tmpl'], input_file, variables=_tmpl_vars)

            if backup:
                backup_filename = _render_template(config['bkup_tmpl'], input_file, variables=_tmpl_vars)

            # check for existing output files
            if os.path.exists(output_filename) and not overwrite:
                progress['exists'] += 1

                exists_error = FileExistsError(f"File {dest!r} (for input {input_file!r}) already exists")
                print_err(exists_error)

                maybe_exit(2)
                continue

            # run differ and change result to be the diff output
            if diff:
                diff_output = list(difflib.context_diff(data.splitlines(True),
                                                        formatted_json.splitlines(True),
                                                        input_file, 'Formatted'))
                result = ''.join(diff_output)

                if diff_output:
                    progress['fail'] += 1
                    print_mesg(result)
                else:
                    progress['pass'] += 1
                    print_mesg(f"File {input_file} is properly formatted.")

            if backup and input_file != STDIN:
                try:
                    shutil.copyfile(input_file, output_filename)
                except Exception as err:
                    print_err(err)

            if output_filename == STDOUT:
                sys.stdout.write(result)
            else:
                with open(output_filename, 'w', encoding='utf8') as f:
                    f.write(result)

            if not diff:
                progress['pass'] += 1

        except maybe_exception as e:
            # tuples (of exception types) are valid in except blocks
            # an empty tuple means catch nothing
            progress['error'] += 1
            print_err(e)
            maybe_exit(2)

        finally:
            progress['remaining'] -= 1

    if parse:
        print_mesg(', '.join(f'{s}: {n}' for s, n in progress.items() if s in ['pass', 'error']))

    elif diff:
        print_mesg(', '.join(f'{s}: {n}' for s, n in progress.items() if s != 'remaining'))

    # leave first two bits for real exit codes and use the rest for failure count
    exit(progress['fail'] << 2)

def parse_arguments(args: argparse.Namespace, parser: argparse.ArgumentParser) -> dict:
    # handle help messages
    if args.long_help:
        parser.epilog = parser.long_epilog
        parser.print_help()
        exit(0)

    elif args.short_help:
        parser.print_help()
        exit(0)

    # helper functions
    def normpath(path):
        if not path:
            return path
        if '~' in path:
            path = os.path.expanduser(path)
        return os.path.normpath(path)

    def outargs(**kwargs):
        out_tmpl = kwargs.get('out_tmpl') or args.output_template or _default_output_template
        bkup_tmpl = kwargs.get('bkup_tmpl') or args.backup_template or _default_backup_template
        return {
            "out": normpath(kwargs.get('out', None)),
            "out_tmpl": normpath(out_tmpl),
            "bkup_tmpl": normpath(bkup_tmpl),
        }

    io_mapping = {}

    # validate formatter and warnings args first
    for arg in args.formatter:
        arg = arg[0]

        if '=' not in arg:
            parser.error(f"Malformed formatter argument: {arg}")

        opt, value = arg.split('=')

        if opt not in _formatting_choices:
            parser.error(f"{opt!r} is not a valid formatter choice (from argument {arg!r})")

    for arg in args.warnings:
        arg = arg[0]

        if '=' not in arg:
            parser.error(f"Malformed warning argument: {arg}")

        opt, value = arg.split('=')

        if opt not in _warning_choices:
            parser.error(f"{opt!r} is not a valid warning choice (from argument {arg!r})")

    ## parse input arguments
    # map stdin
    if args.input == ['-']:
        io_mapping['-'] = outargs()

    # expand globs
    elif args.glob:
        for input_arg in args.input:
            for f in glob.glob(input_arg):
                io_mapping[normpath(f)] = outargs()

    # normal file mapping
    else:
        io_mapping = {normpath(f): outargs() for f in args.input}

    ## parse output arguments
    # map stdout
    if args.output == ['-']:
        for f, opts in io_mapping.items():
            opts.update(outargs(out='-'))

    # check for input/output mismatch
    elif len(args.input) != len(args.output) and args.output and not args.output_template:
        parser.error(f"Number of inputs ({len(args.input)}) does not match outputs ({len(args.output)})")

    # set output to input when overwriting
    elif args.overwrite and not args.output:
        for input_file, opts in io_mapping.items():
            opts.update(outargs(out=input_file))

    # normal output mapping
    else:
        for opts, out in zip(io_mapping.values(), args.output):
            opts.update(outargs(out=out))

    return io_mapping


if __name__ == '__main__':
    _file_ = os.path.basename(__file__)
    _warning_arg_list = '\n'.join(textwrap.wrap(', '.join(_warning_choices), 78,
                                                break_on_hyphens=False,
                                                subsequent_indent=' '*6))
    long_epilog = f"""\

Template variables
===================

  Template variables are indicated using python new-style formatting syntax
  using curly braces.  Consequently, the python format-mini language is
  supported during the template rendering.  This is most useful for numeric
  and datetime variables.  This option can be used to route output files
  into specific folders or for name sorting purposes.  For example:

    {_file_} -B "{{parent}}/{{unixtime:.0f}}/{{input_pos}}_{{filename}}" ...

  bob@foo:~/cdda/data$ python3 tools/{_file_} -i json/skills.json ...

  |   variable  |   type   |         example data         | note |
  |-------------|----------|------------------------------|------|
  | filename    | string   | skills.json                  |      |
  | stem        | string   | skills                       |      |
  | ext         | string   | json                         |      |
  | suffix      | string   | .json                        |      |
  | path        | string   | json/skills.json             | [1]  |
  | abs_path    | string   | /home/(...)/json/skills.json | [2]  |
  | parent      | string   | /home/bob/cdda/data/json     |      |
  | sep         | string   | / (on nix)  \\ (on win)      |      |
  | cwd         | string   | /home/bob/cdda/data          |      |
  | home        | string   | /home/bob                    | [3]  |
  | stat        | stat     | (see note below)             | [4]  |
  | unixtime    | float    | 1581670653.0311236           | [5]  |
  | datetime    | datetime | 2020-02-14_03.57.33          | [6]  |
  | utcdatetime | datetime | (same as above)              | [7]  |
  | input_pos   | int      | 1                            | [8]  |
  | input_total | int      | 26                           | [9]  |

  [1] same as input
  [2] absolute path of input file
  [3] same as $HOME or %HOMEPATH%
  [4] an os.stat_result object, access the stat struct fields via
      dot access, i.e. {{stat.st_size}}
  [5] use a precision specifier for an int value, i.e. {{unixtime:.0f}}
  [6] a datetime.datetime object, use strftime variables, i.e.
      {{:%Y-%m-%d_%H.%M.%S}}
  [7] same as above, but in UTC instead of local time
  [8] position of file in input queue
  [9] total number of inputs in queue


Formatter arguments
====================

  -F/--formatter arguments are passed in 'arg=value' format, i.e.

    {_file_} -F cast-ints=true -i ...

  These options will change the formatting rules and alter the output
  based on the given values.  The default values represent their type
  and their behaviour when "disabled" (not passed).

  indent-width=2
    Sets the size of indentation levels

  max-line-length=120
    Sets the maximum length containers can be before they're wrapped

  wrap-overrides=rows,blueprint
    Comma separated list of key names to always wrap if their values
    are collections and have at least one item

  no-wrap-overrides=false
    Disables the behavior of wrap-overrides

  auto-split-stringarray-lines=false
    Automatically split long strings in arrays containing a single string

  always-wrap-depth=1
    Unconditionally wrap containers at this depth regardless of contents

  cast-ints=true
    Cast integers to c_longlong to simulate integer wraparound

  round-floats=true
    Round floats to 6 decimal places


Warning arguments
==================

  This feature is not yet implemented.

"""
    # TODO: move to epilog
    # all:
    #   enable all warnings
    #
    # stringarray-spaces:
    #   warn if there is no space (without period) at the end of a string in a stringarray
    #
    # long-list-item:
    #   warn if a list with a single item is longer than the length limit (may be a stringarray)
    #
    # int-wraparound:
    #   warn if an int may overflow or underflow when cast to longlong (int64)
    #
    # float-rounding:
    #   warn if a float has more than 6 decimal places of precision

    parser = argparse.ArgumentParser(prog=_file_, add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Reformat json to comply with C:DDA formatting standards. "
                    "Run with --help for more information.")

    parser.long_epilog = long_epilog

    # short help
    parser.add_argument('-h', action='store_true', dest='short_help',
                        help="show a short help message and exit")

    # long help
    parser.add_argument('--help', action='store_true', dest='long_help',
                        help="show a long help message and exit")

    # edit in place (atomic replace)
    parser.add_argument('-r', '--overwrite', action='store_true',
                        help="write output to the source file (backup files will always be overwritten")

    # keep original as copy/backup
    parser.add_argument('-b', '--backup', action='store_true',
                        help="keep a backup copy of the inputs")

    # backup name template
    parser.add_argument('-B', '--backup-template',
                        help="format template for backup filenames; see --help for variables")

    # debug logging (change to loglevel style?)
    parser.add_argument('-V', '--verbose', action='store_true',
                        help="print debug information to stderr")

    # quiet output
    parser.add_argument('-q', '--quiet', action='store_true',
                        help="do not print any extraneous messages")

    # ignore read errors from multiple inputs? -c, --continue
    parser.add_argument('-c', '--continue', action='store_true', dest='continue_on_err',
                        help="continue processing inputs when an error is encountered")

    # parse input, only checking for syntax errors
    parser.add_argument('-p', '--parse', action='store_true',
                        help="only check input for syntax errors")

    # parse input, reformats it, and checks for diffs
    parser.add_argument('-d', '--diff', action='store_true',
                        help="Verify that the input json matches the formatted output json. "
                             "Instead of formatted json being written to output files, a "
                             "diff is generated and written instead; implies -c")

    # show/hide warnings i.e. -W all
    parser.add_argument('-W', '--warnings', action='append', nargs=1, default=[],
                        # choices=_warning_choices,
                        help="emit specified warnings during formatting for potential issues; see --help")

    # formatting options
    parser.add_argument('-F', '--formatter', action='append', nargs='+', default=[],
                        # choices=_formatting_choices,
                        help="enable specific formatting tweaks; see --help")

    # glob input flag
    parser.add_argument('-g', '--glob', action='store_true',
                        help="use glob matching for input parsing")

    # input filename(s), read from stdin with `-`
    parser.add_argument('-i', dest='input', nargs='+',
                        help="one or more filenames to read from; pass - to read from stdin")

    # output name template (%n_new.json) etc
    parser.add_argument('-T', '--output-template',
                        help="format template for ouput filenames; see --help for variables")

    # output filename, output to stdout with `-`
    # if taking multiple inputs and input/output counts don't match: error
    parser.add_argument('-o', dest='output', nargs='*', default=[],
                        help="Zero or more filenames to write to. The number of output arguments "
                             "must match the number of input arguments if there is more than one "
                             "input. This option is not required when using an output template or "
                             "overwrite mode. Pass - to write to stdout")

    argv = sys.argv[1:]

    try:
        args = parser.parse_args(argv)
        io_mapping = parse_arguments(args, parser)
    except SystemExit as e:
        e.code = e.code and 1 # short circuit on falsy trick
        raise e from None

    main(io_mapping, **vars(args))
