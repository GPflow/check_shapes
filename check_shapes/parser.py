# Copyright 2022 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=unused-argument

import re
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from .argument_ref import (
    RESULT_TOKEN,
    AllElementsRef,
    ArgumentRef,
    AttributeArgumentRef,
    IndexArgumentRef,
    KeysRef,
    RootArgumentRef,
    ValuesRef,
)
from .bool_specs import (
    BoolTest,
    ParsedAndBoolSpec,
    ParsedArgumentRefBoolSpec,
    ParsedBoolSpec,
    ParsedNotBoolSpec,
    ParsedOrBoolSpec,
)
from .config import DocstringFormat, get_rewrite_docstrings
from .error_contexts import (
    ArgumentContext,
    ErrorContext,
    MultipleElementBoolContext,
    StackContext,
    UnexpectedInputContext,
)
from .exceptions import SpecificationParseError
from .specs import (
    ParsedArgumentSpec,
    ParsedDimensionSpec,
    ParsedFunctionSpec,
    ParsedNoteSpec,
    ParsedShapeSpec,
    ParsedTensorSpec,
)

_TPattern = Tuple[str, str]
"""
A terminal kind in the ``check_shapes`` specification language.
"""


def _literal_terminal(literal: str) -> _TPattern:
    """
    Create a terminal that only matches exactly the given ``literal``.
    """
    return re.escape(literal), f'"{literal}"'


_T_STAR = _literal_terminal("*")
_T_ELLIPSIS = _literal_terminal("...")
_T_DOT = _literal_terminal(".")
_T_COLON = _literal_terminal(":")
_T_COMMA = _literal_terminal(",")
_T_LEFT_SQ = _literal_terminal("[")
_T_RIGHT_SQ = _literal_terminal("]")
_T_LEFT_PAREN = _literal_terminal("(")
_T_RIGHT_PAREN = _literal_terminal(")")
_T_BROADCAST = _literal_terminal("broadcast")
_T_NONE = _literal_terminal("None")
_T_IF = _literal_terminal("if")
_T_IS = _literal_terminal("is")
_T_NOT = _literal_terminal("not")
_T_AND = _literal_terminal("and")
_T_OR = _literal_terminal("or")
_T_INT = (r"[0-9]+", "<integer>")
_T_NAME = (r"[_a-zA-Z][_a-zA-Z0-9]*", "<variable name>")
_T_NOTE = (r"#.*", '"#" <note>')
_T_IGNORE = (r"\s+", "<ignoreable whitespace>")

_T_ERROR = (r"", "<unable to tokenize>")
"""
Pseudo-terminal that is never actually matched against, but is used to represent an error during
tokenisation.
"""

_TERMINALS = (
    _T_STAR,
    _T_ELLIPSIS,
    _T_DOT,
    _T_COLON,
    _T_COMMA,
    _T_LEFT_SQ,
    _T_RIGHT_SQ,
    _T_LEFT_PAREN,
    _T_RIGHT_PAREN,
    _T_BROADCAST,
    _T_NONE,
    _T_IF,
    _T_IS,
    _T_NOT,
    _T_AND,
    _T_OR,
    _T_INT,
    _T_NAME,
    _T_NOTE,
    _T_IGNORE,
)

_TERMINALS_RE = re.compile("|".join(f"({t[0]})" for t in _TERMINALS), flags=re.DOTALL)
_SPACES_RE = re.compile(r"\s+")


class _ShapeSpecParser:
    """
    Recursive descent parser for `check_shapes` specifications.

    The terminal kinds of the language are defined by the above ``_T_*`` variables and
    ``_TERMINALS``. Each of those are a 2-tuple with:

    1. The regular expression pattern to match.
    2. A human-readable name of the pattern, that will be used in error messages.

    The ``__init__`` below tokenises the input ``source`` into ``self.tokens``, which is a list of
    3-tuples with:

    1. The index into ``self.source`` where the token was matched.
    2. The kind of token, represented by the ``_T_*`` pattern that matched.
    3. The content of the token - the substring of ``self.source`` that was matched.

    ``self.i`` holds the input of the next token to process, so the next token is
    ``self.tokens[self.i]``.

    To process the next token call ``self._accept`` with the expected token kind.
    If the next token fails to match expectations the expectation is added to ``self.expected``, and
    an error code is returned. If the token is accepted, ``self.expected`` is cleared.
    If we give up matching the next token, we call ``self._unexpected_input`` which will use
    ``self.expected`` to create an error messages with all the token kinds we failed to match.

    Otherwise this class has a method for each non-terminal / construct of this language.
    After parsing your specification, call ``self.close()`` to check that the entire input string
    has been consumed.
    """

    def __init__(self, source: str, context: ErrorContext) -> None:
        l = len(source)
        i = 0
        tokens: List[Tuple[int, _TPattern, str]] = []
        while i < l:
            match = _TERMINALS_RE.match(source, i)
            if match:
                terminal = _TERMINALS[match.lastindex - 1]  # type: ignore[operator]
                if terminal is not _T_IGNORE:
                    tokens.append((i, terminal, match.group()))
                i = match.end()
            else:
                tokens.append((i, _T_ERROR, ""))
                i = l

        self.source = source
        self.tokens = tokens
        self.i = 0
        self.context = context
        self.expected: List[str] = []

    def _unexpected_input(self) -> None:
        """
        Call this when we give up on consuming the next token.
        """
        l = len(self.tokens)
        is_eof = self.i >= l
        i = len(self.source) if is_eof else self.tokens[self.i][0]
        context = StackContext(
            self.context, UnexpectedInputContext(self.source, i, tuple(self.expected), is_eof)
        )
        raise SpecificationParseError(context)

    def _multiple_element_spec(self, is_for_bool_spec: bool) -> None:
        """
        Call this if the user tries to use a multi-element argument specification in a boolean
        context.
        """
        if not is_for_bool_spec:
            return
        l = len(self.tokens)
        token_i = self.i - 1
        is_eof = token_i >= l
        source_i = len(self.source) if is_eof else self.tokens[token_i][0]
        context = StackContext(self.context, MultipleElementBoolContext(self.source, source_i))
        raise SpecificationParseError(context)

    def _accept(
        self,
        expected: _TPattern,
        expected_value: Optional[str] = None,
        *,
        required: bool = False,
        out_type: Optional[Type[Any]] = None,
        error_mesg: Optional[str] = None,
    ) -> Any:
        """
        Tries to consume the next token.

        The token is only consumed if it matches certain expectations. If not, a value indicating
        error is returned.

        For the token to be consumed it must have ``expected`` kind. If ``expected_value`` is set
        the value of the token must also match this value.

        * If ``required=True`` and the token is not consumed, then an error is raised.
        * If ``out_type`` is set and the token is consumed then the value of the token is converted
          to the given type and returned.
        * If ``out_type`` is set and the token is is not consumed then ``None`` is returned.
        * If ``required=False`` and ``out_type=None`` and then a ``bool`` indicating whether the
          token was consumed is returned.

        ``error_mesg`` can be used to define the user-visible "expected" value to print if a token
        is not found. If unset, a default value for the kind of token is used.
        """
        if error_mesg is None:
            error_mesg = expected[1]
        self.expected.append(error_mesg)

        if self.i >= len(self.tokens):
            if required:
                self._unexpected_input()
            return None if out_type else False
        token = self.tokens[self.i]
        if token[1] is expected and (expected_value is None or token[2] == expected_value):
            self.expected.clear()
            self.i += 1
            return out_type(token[2]) if out_type else True
        if required:
            self._unexpected_input()
        return None if out_type else False

    def _note_spec(self) -> Optional[ParsedNoteSpec]:
        value = self._accept(_T_NOTE, out_type=str)
        if value is None:
            return None
        value = value[1:]
        value = value.strip()
        value = _SPACES_RE.sub(" ", value)
        return ParsedNoteSpec(value)

    def _dimension_spec(self) -> ParsedDimensionSpec:
        match = False
        constant: Optional[int] = None
        variable_name: Optional[str] = None
        variable_rank = False
        broadcastable = False
        if self._accept(_T_BROADCAST):
            broadcastable = True
        if self._accept(_T_ELLIPSIS):
            variable_rank = True
            match = True
        elif self._accept(_T_DOT) or self._accept(_T_NONE):
            match = True
        else:
            int_value = self._accept(_T_INT, out_type=int)
            if int_value is not None:
                constant = int_value
                match = True
            else:
                if self._accept(_T_STAR):
                    variable_rank = True
                    match = True
                name_value = self._accept(_T_NAME, out_type=str)
                if name_value is not None:
                    variable_name = name_value
                    match = True
                if not variable_rank and self._accept(_T_ELLIPSIS):
                    variable_rank = True
                    match = True
        if not match:
            self._unexpected_input()
            assert False, "This line should be unreachable."
        return ParsedDimensionSpec(
            constant=constant,
            variable_name=variable_name,
            variable_rank=variable_rank,
            broadcastable=broadcastable,
        )

    def _shape_spec(self) -> ParsedShapeSpec:
        self._accept(_T_LEFT_SQ, required=True)
        dimension_specs = []
        if not self._accept(_T_RIGHT_SQ):
            while True:
                dimension_specs.append(self._dimension_spec())
                if self._accept(_T_COMMA):
                    if self._accept(_T_RIGHT_SQ):
                        break
                elif self._accept(_T_RIGHT_SQ):
                    break
                else:
                    self._unexpected_input()
        return ParsedShapeSpec(tuple(dimension_specs))

    def tensor_spec(self) -> ParsedTensorSpec:
        shape_spec = self._shape_spec()
        note_spec = self._note_spec()
        return ParsedTensorSpec(shape_spec, note_spec)

    def _argument_ref(self, is_for_bool_spec: bool) -> Optional[ArgumentRef]:
        name_value = self._accept(_T_NAME, out_type=str)
        if name_value is None:
            return None
        result: ArgumentRef = RootArgumentRef(name_value)
        while True:
            if self._accept(_T_DOT):
                if self._accept(_T_NAME, "keys", error_mesg='"keys"'):
                    self._multiple_element_spec(is_for_bool_spec)
                    self._accept(_T_LEFT_PAREN, required=True)
                    self._accept(_T_RIGHT_PAREN, required=True)
                    result = KeysRef(result)
                elif self._accept(_T_NAME, "values", error_mesg='"values"'):
                    self._multiple_element_spec(is_for_bool_spec)
                    self._accept(_T_LEFT_PAREN, required=True)
                    self._accept(_T_RIGHT_PAREN, required=True)
                    result = ValuesRef(result)
                else:
                    name_value = self._accept(_T_NAME, out_type=str)
                    if name_value is None:
                        self._unexpected_input()
                        assert False, "This line should be unreachable."
                    result = AttributeArgumentRef(result, name_value)
            elif self._accept(_T_LEFT_SQ):
                if self._accept(_T_NAME, "all", error_mesg='"all"'):
                    self._multiple_element_spec(is_for_bool_spec)
                    result = AllElementsRef(result)
                else:
                    int_value = self._accept(_T_INT, out_type=int)
                    if int_value is None:
                        self._unexpected_input()
                        assert False, "This line should be unreachable."
                    result = IndexArgumentRef(result, int_value)
                self._accept(_T_RIGHT_SQ, required=True)
            else:
                break
        return result

    def _bool_spec_base(self) -> ParsedBoolSpec:
        if self._accept(_T_LEFT_PAREN):
            bool_spec: ParsedBoolSpec = self._bool_spec()
            self._accept(_T_RIGHT_PAREN, required=True)
            return bool_spec
        if self._accept(_T_NOT):
            return ParsedNotBoolSpec(self._bool_spec_base())
        maybe_argument_ref = self._argument_ref(is_for_bool_spec=True)
        if maybe_argument_ref is None:
            self._unexpected_input()
        else:
            argument_ref = maybe_argument_ref
        if self._accept(_T_IS):
            if self._accept(_T_NONE):
                bool_spec = ParsedArgumentRefBoolSpec(argument_ref, BoolTest.IS_NONE)
            elif self._accept(_T_NOT):
                self._accept(_T_NONE, required=True)
                bool_spec = ParsedArgumentRefBoolSpec(argument_ref, BoolTest.IS_NOT_NONE)
            else:
                self._unexpected_input()
        else:
            bool_spec = ParsedArgumentRefBoolSpec(argument_ref, BoolTest.BOOL)
        return bool_spec

    def _bool_spec_and(self) -> ParsedBoolSpec:
        lhs = self._bool_spec_base()
        if self._accept(_T_AND):
            rhs = self._bool_spec_and()
            return ParsedAndBoolSpec(lhs, rhs)
        return lhs

    def _bool_spec_or(self) -> ParsedBoolSpec:
        lhs = self._bool_spec_and()
        if self._accept(_T_OR):
            rhs = self._bool_spec_or()
            return ParsedOrBoolSpec(lhs, rhs)
        return lhs

    def _bool_spec(self) -> ParsedBoolSpec:
        return self._bool_spec_or()

    def _argument_spec(self) -> Optional[ParsedArgumentSpec]:
        argument_ref = self._argument_ref(is_for_bool_spec=False)
        if argument_ref is None:
            return None
        self._accept(_T_COLON, required=True)
        shape_spec = self._shape_spec()
        bool_spec = self._bool_spec() if self._accept(_T_IF) else None
        note_spec = self._note_spec()
        return ParsedArgumentSpec(
            argument_ref,
            ParsedTensorSpec(shape_spec, note_spec),
            bool_spec,
        )

    def argument_or_note_spec(self) -> Union[ParsedNoteSpec, ParsedArgumentSpec]:
        argument_spec = self._argument_spec()
        if argument_spec is not None:
            return argument_spec
        note_spec = self._note_spec()
        if note_spec is not None:
            return note_spec
        self._unexpected_input()
        assert False, "This line should be unreachable."

    def close(self) -> None:
        if self.i != len(self.tokens):
            self.expected.append("<end of input>")
            self._unexpected_input()

    def __str__(self) -> str:
        tokens = []
        for i, t in enumerate(self.tokens):
            prefix = ">>> " if i == self.i else "    "
            tokens.append(prefix + str(t))
        return "\n".join(tokens)


_InfoField = Tuple[str, str, int, int]

_PARAM_FIELDS = {"param", "parameter", "arg", "argument", "key", "keyword"}
_TYPE_FIELDS = {"type"}
_RAISES_FIELDS = {"raises", "raise", "except", "exception"}
_VAR_FIELDS = {"var", "ivar", "cvar"}
_VARTYPE_FIELDS = {"vartype"}
_RETURNS_FIELDS = {"returns", "return"}
_RTYPE_FIELDS = {"rtype"}
_META_FIELDS = {"meta"}
_ALL_FIELDS = (
    _PARAM_FIELDS
    | _TYPE_FIELDS
    | _RAISES_FIELDS
    | _VAR_FIELDS
    | _VARTYPE_FIELDS
    | _RETURNS_FIELDS
    | _RTYPE_FIELDS
    | _META_FIELDS
)
_INFO_FIELD_RE = re.compile(
    "^ *(:(" + "|".join(_ALL_FIELDS) + ")( +([ _a-zA-Z0-9]+))? *: *)", flags=re.MULTILINE
)
_TEXT_RE = re.compile(r"\s*(.*\S)\s*", flags=re.DOTALL)


class _RewritedocString:
    def __init__(self, source: str, function_spec: ParsedFunctionSpec) -> None:
        self._source = source
        self._spec_lines = self._argument_specs_to_sphinx(function_spec.arguments)
        self._notes = tuple(note.note for note in function_spec.notes)
        self._indent = self._guess_indent(source)

    def _argument_specs_to_sphinx(
        self,
        argument_specs: Collection[ParsedArgumentSpec],
    ) -> Mapping[str, Sequence[str]]:
        result: Dict[str, List[str]] = {}
        for spec in argument_specs:
            result.setdefault(spec.argument_ref.root_argument_name, []).append(
                self._argument_spec_to_sphinx(spec)
            )
        for lines in result.values():
            lines.sort()
        return result

    def _argument_spec_to_sphinx(self, argument_spec: ParsedArgumentSpec) -> str:
        tensor_spec = argument_spec.tensor
        shape_spec = tensor_spec.shape
        out = []
        out.append(f"* **{repr(argument_spec.argument_ref)}**")
        out.append(" has shape [")
        out.append(self._shape_spec_to_sphinx(shape_spec))
        out.append("]")

        if argument_spec.condition is not None:
            out.append(" if ")
            out.append(self._bool_spec_to_sphinx(argument_spec.condition, False))

        out.append(".")

        if tensor_spec.note is not None:
            note_spec = tensor_spec.note
            out.append(" ")
            out.append(note_spec.note)
        return "".join(out)

    def _bool_spec_to_sphinx(self, bool_spec: ParsedBoolSpec, paren_wrap: bool) -> str:
        if isinstance(bool_spec, ParsedOrBoolSpec):
            result = (
                self._bool_spec_to_sphinx(bool_spec.left, True)
                + " or "
                + self._bool_spec_to_sphinx(bool_spec.right, True)
            )
        elif isinstance(bool_spec, ParsedAndBoolSpec):
            result = (
                self._bool_spec_to_sphinx(bool_spec.left, True)
                + " and "
                + self._bool_spec_to_sphinx(bool_spec.right, True)
            )
        elif isinstance(bool_spec, ParsedNotBoolSpec):
            result = "not " + self._bool_spec_to_sphinx(bool_spec.right, True)
        else:
            assert isinstance(bool_spec, ParsedArgumentRefBoolSpec)
            if bool_spec.bool_test == BoolTest.BOOL:
                paren_wrap = False  # Never wrap a stand-alone argument.
                result = f"*{bool_spec.argument_ref!r}*"
            elif bool_spec.bool_test == BoolTest.IS_NONE:
                result = f"*{bool_spec.argument_ref!r}* is *None*"
            else:
                assert bool_spec.bool_test == BoolTest.IS_NOT_NONE
                result = f"*{bool_spec.argument_ref!r}* is not *None*"

        if paren_wrap:
            result = f"({result})"

        return result

    def _shape_spec_to_sphinx(self, shape_spec: ParsedShapeSpec) -> str:
        return ", ".join(self._dim_spec_to_sphinx(dim) for dim in shape_spec.dims)

    def _dim_spec_to_sphinx(self, dim_spec: ParsedDimensionSpec) -> str:
        tokens = []

        if dim_spec.broadcastable:
            tokens.append("broadcast ")

        if dim_spec.constant is not None:
            tokens.append(str(dim_spec.constant))
        elif dim_spec.variable_name:
            tokens.append(f"*{dim_spec.variable_name}*")
        else:
            if not dim_spec.variable_rank:
                tokens.append(".")

        if dim_spec.variable_rank:
            tokens.append("...")

        return "".join(tokens)

    def _guess_indent(self, docstring: str) -> Optional[int]:
        """
        Infer the level of indentation of a docstring.

        Returns `None` if the indentation could not be inferred.
        """
        # Algorithm adapted from:
        #     https://www.python.org/dev/peps/pep-0257/#handling-docstring-indentation

        # Convert tabs to spaces (following the normal Python rules)
        # and split into a list of lines:
        lines = docstring.expandtabs().splitlines()
        # Determine minimum indentation (first line doesn't count):
        no_indent = -1
        indent = no_indent
        for line in lines[1:]:
            stripped = line.lstrip()
            if not stripped:
                continue
            line_indent = len(line) - len(stripped)
            if indent == no_indent or line_indent < indent:
                indent = line_indent
        return indent if indent != no_indent else None

    def _insert_spec_lines(
        self, out: List[str], pos: int, spec_lines: Sequence[str], info_field: _InfoField
    ) -> int:
        leading_str = self._source[pos : info_field[2]].rstrip()
        docs_start = pos + len(leading_str)
        docs_str = self._source[docs_start : info_field[3]]
        trailing_str = docs_str.lstrip()

        docs_indent = self._guess_indent(docs_str)
        if docs_indent is None:
            if self._indent is None:
                docs_indent = 4
            else:
                docs_indent = self._indent + 4
        indent_str = "\n" + docs_indent * " "

        out.append(leading_str)
        for spec_line in spec_lines:
            out.append(indent_str)
            out.append(spec_line)
        out.append("\n")
        out.append(indent_str)
        out.append(trailing_str)
        return info_field[3]

    def _insert_param_info_fields(
        self,
        is_first_info_field: bool,
        spec_lines: Mapping[str, Sequence[str]],
        out: List[str],
        pos: int,
    ) -> int:
        leading_str = self._source[pos:].rstrip()
        out.append(leading_str)
        pos += len(leading_str)

        if not self._source:
            # Case where nothing preceeds these fields. Just write them.
            needed_newlines = 0
        elif is_first_info_field:
            # Free-form documentation preceeds these fields. Have 2 newlines to separate them.
            needed_newlines = 2
        else:
            # Another info-field preceeds these fields.
            needed_newlines = 1

        indent = self._indent or 0
        indent_str = indent * " "
        indent_one_str = 4 * " "

        for arg_name, arg_lines in spec_lines.items():
            out.append(needed_newlines * "\n")
            needed_newlines = 1

            out.append(indent_str)
            if arg_name == RESULT_TOKEN:
                out.append(":returns:")
            else:
                out.append(f":param {arg_name}:")
            for arg_line in arg_lines:
                out.append("\n")
                out.append(indent_str)
                out.append(indent_one_str)
                out.append(arg_line)

        return pos

    def docstring(self) -> str:
        # The strategy here is:
        # * `out` contains a list of strings that will be concatenated and form the final result.
        # * `pos` is the position such that `self._source[:pos]` has already been added to `out`,
        #   and `self._source[pos:]` still needs to be added.
        # * When visiting children we pass `out` and `pos`, and the children add content to `out`
        #   and return a new `pos`.
        out: List[str] = []
        info_fields: List[_InfoField] = []
        pos = 0

        def get_text_indices(start: int, end: int) -> Tuple[int, int]:
            match = _TEXT_RE.fullmatch(self._source, start, end)
            if match:
                return match.start(1), match.end(1)
            else:
                return start, start

        prev_match = None
        for match in _INFO_FIELD_RE.finditer(self._source):
            if prev_match is None:
                docs_end = get_text_indices(0, match.start(0))[1]
            else:
                info_fields.append(
                    (
                        prev_match.group(2),
                        prev_match.group(4),
                        *get_text_indices(prev_match.end(0), match.start(1)),
                    )
                )

            prev_match = match

        if prev_match is None:
            docs_match = _TEXT_RE.match(self._source)
            docs_end = docs_match.end(0) if docs_match else 0
        else:
            info_fields.append(
                (
                    prev_match.group(2),
                    prev_match.group(4),
                    *get_text_indices(prev_match.end(0), len(self._source)),
                )
            )

        if self._notes:
            if docs_end:
                out.append(self._source[pos:docs_end])
                pos = docs_end
            indent = self._indent or 0
            indent_str = indent * " "
            for note in self._notes:
                if out:
                    out.append("\n\n")
                out.append(indent_str)
                out.append(note)

        pos = self._info_fields(info_fields, out, pos)
        out.append(self._source[pos:])

        return "".join(out)

    def _info_fields(self, info_fields: Sequence[_InfoField], out: List[str], pos: int) -> int:
        spec_lines = dict(self._spec_lines)
        is_first_info_field = True
        for info_field in info_fields:
            # This will remove the self._spec_lines corresponding to found `:param:`'s.
            if info_field[0] in _PARAM_FIELDS:
                pos = self._info_field_param(info_field, spec_lines, out, pos)
            elif info_field[0] in _RETURNS_FIELDS:
                pos = self._info_field_returns(info_field, spec_lines, out, pos)
            is_first_info_field = False

        # Add any remaining `:param:`s:
        pos = self._insert_param_info_fields(is_first_info_field, spec_lines, out, pos)

        # Make sure info fields are terminated by a new-line:
        if self._spec_lines:
            if (pos >= len(self._source)) or (self._source[pos] != "\n"):
                out.append("\n")

        return pos

    def _info_field_param(
        self, info_field: _InfoField, spec_lines: Dict[str, Sequence[str]], out: List[str], pos: int
    ) -> int:
        arg_name = info_field[1]
        arg_lines = spec_lines.pop(arg_name, None)
        if arg_lines:
            pos = self._insert_spec_lines(out, pos, arg_lines, info_field)
        return pos

    def _info_field_returns(
        self, info_field: _InfoField, spec_lines: Dict[str, Sequence[str]], out: List[str], pos: int
    ) -> int:
        return_lines = spec_lines.pop(RESULT_TOKEN, None)
        if return_lines:
            pos = self._insert_spec_lines(out, pos, return_lines, info_field)
        return pos


_TENSOR_SPEC_CACHE: Dict[str, ParsedTensorSpec] = {}
_ARGUMENT_OR_NOTE_SPEC_CACHE: Dict[str, Union[ParsedArgumentSpec, ParsedNoteSpec]] = {}
_SPHINX_REWRITE_CACHE: Dict[Tuple[str, ParsedFunctionSpec], str] = {}


def parse_tensor_spec(tensor_spec: str, context: ErrorContext) -> ParsedTensorSpec:
    """
    Parse a `check_shapes` tensor specification.
    """
    result = _TENSOR_SPEC_CACHE.get(tensor_spec)
    if result is None:
        parser = _ShapeSpecParser(tensor_spec, context)
        result = parser.tensor_spec()
        parser.close()
        _TENSOR_SPEC_CACHE[tensor_spec] = result
    return result


def parse_function_spec(function_spec: Sequence[str], context: ErrorContext) -> ParsedFunctionSpec:
    """
    Parse all `check_shapes` argument or note specification for a single function.
    """
    arguments = []
    notes = []
    for i, spec in enumerate(function_spec):
        result = _ARGUMENT_OR_NOTE_SPEC_CACHE.get(spec)
        if result is None:
            argument_context = StackContext(context, ArgumentContext(i))
            parser = _ShapeSpecParser(spec, argument_context)
            result = parser.argument_or_note_spec()
            parser.close()
            _ARGUMENT_OR_NOTE_SPEC_CACHE[spec] = result
        if isinstance(result, ParsedArgumentSpec):
            arguments.append(result)
        else:
            assert isinstance(result, ParsedNoteSpec)
            notes.append(result)
    return ParsedFunctionSpec(tuple(arguments), tuple(notes))


def parse_and_rewrite_docstring(
    docstring: Optional[str], function_spec: ParsedFunctionSpec, context: ErrorContext
) -> Optional[str]:
    """
    Rewrite `docstring` to include the shapes specified by the `argument_specs`.
    """
    if docstring is None:
        return None

    docstring_format = get_rewrite_docstrings()
    if docstring_format == DocstringFormat.NONE:
        return docstring

    assert docstring_format == DocstringFormat.SPHINX, (
        f"Current docstring format is {docstring_format}, but I don't know how to rewrite that."
        " See `check_shapes.config.set_rewrite_docstrings`."
    )

    result = _SPHINX_REWRITE_CACHE.get((docstring, function_spec))
    if result is None:
        parser = _RewritedocString(docstring, function_spec)
        result = parser.docstring()
        _SPHINX_REWRITE_CACHE[(docstring, function_spec)] = result
    return result
