# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from .._lazy_module import _LazyModule


_import_structure = {
    "callbacks": [
        "BEMACallback",
        "LogCompletionsCallback",
        "RichProgressCallback",
        "SyncRefModelCallback",
        "WeaveCallback",
    ],
    "model_config": ["ModelConfig"],
    "sft_config": ["SFTConfig"],
    "sft_trainer": ["SFTTrainer"],
    "utils": [
        "disable_dropout_in_model",
        "ensure_master_addr_port",
        "get_kbit_device_map",
        "get_peft_config",
        "get_quantization_config",
        "patch_chunked_lm_head",
    ],
}

if TYPE_CHECKING:
    from .callbacks import (
        BEMACallback,
        LogCompletionsCallback,
        RichProgressCallback,
        SyncRefModelCallback,
        WeaveCallback,
    )
    from .model_config import ModelConfig
    from .sft_config import SFTConfig
    from .sft_trainer import SFTTrainer
    from .utils import (
        disable_dropout_in_model,
        ensure_master_addr_port,
        get_kbit_device_map,
        get_peft_config,
        get_quantization_config,
        patch_chunked_lm_head,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
