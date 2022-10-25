from typing import List, Dict, Union, Any
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict, Literal, overload


class PipeOutputElement(TypedDict):
    input: str
    output: Dict[str, Any]
    modified_input: str


class TrainDataItem(TypedDict):
    question: str
    context: str
    native_context: str


PipeOutput = Union[List[PipeOutputElement], PipeOutputElement]
TrainData = List[TrainDataItem]
