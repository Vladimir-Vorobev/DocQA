from typing import List, Dict, Union, Any, TypedDict


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
