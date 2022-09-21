from typing import List, Dict, Union, Any, TypedDict


class PipeOutputElement(TypedDict):
    input: str
    output: Dict[str, Any]
    modified_input: str


PipeOutput = Union[List[PipeOutputElement], PipeOutputElement]
