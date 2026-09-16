// Copyright 2026 Dimensional Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// dimos glue around pointlio-core.
pub mod module;

// The Livox wire protocol, frame assembly and pcap reader, copied verbatim from
// `dimos/hardware/sensors/lidar/livox/rust/src/`. Copied rather than depended on:
// a native module may only reach the shared `native/rust` crates, and livox is a
// specific sensor rather than part of that generic API. Keep these three files
// byte-identical to livox's so a protocol fix can be moved across with `cp`.
#[allow(dead_code)]
pub mod pcap;
#[allow(dead_code)]
pub mod pipeline;
#[allow(dead_code)]
pub mod wire;
