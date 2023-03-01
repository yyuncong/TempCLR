# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .task import *
from .retritask import *
from .alignretritask import *
from .alignlocaltask import *

try:
    from .fairseqmmtask import *
except ImportError:
    pass

