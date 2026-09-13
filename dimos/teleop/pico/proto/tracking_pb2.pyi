from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VRPid(_message.Message):
    __slots__ = ("pid",)
    PID_FIELD_NUMBER: _ClassVar[int]
    pid: int
    def __init__(self, pid: _Optional[int] = ...) -> None: ...

class DeviceStatus(_message.Message):
    __slots__ = ("devid", "status")
    DEVID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    devid: str
    status: int
    def __init__(self, devid: _Optional[str] = ..., status: _Optional[int] = ...) -> None: ...

class DeviceStateJson(_message.Message):
    __slots__ = ("devid", "statejson")
    DEVID_FIELD_NUMBER: _ClassVar[int]
    STATEJSON_FIELD_NUMBER: _ClassVar[int]
    devid: str
    statejson: str
    def __init__(self, devid: _Optional[str] = ..., statejson: _Optional[str] = ...) -> None: ...

class ServerFeedback(_message.Message):
    __slots__ = ("name", "devid", "devstatus", "devicestatejson")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DEVID_FIELD_NUMBER: _ClassVar[int]
    DEVSTATUS_FIELD_NUMBER: _ClassVar[int]
    DEVICESTATEJSON_FIELD_NUMBER: _ClassVar[int]
    name: str
    devid: str
    devstatus: DeviceStatus
    devicestatejson: DeviceStateJson
    def __init__(self, name: _Optional[str] = ..., devid: _Optional[str] = ..., devstatus: _Optional[_Union[DeviceStatus, _Mapping]] = ..., devicestatejson: _Optional[_Union[DeviceStateJson, _Mapping]] = ...) -> None: ...
