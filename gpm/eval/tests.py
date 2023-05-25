# Copyright 2023 Jean-Yves Franceschi, Mike Gartrell, Ludovic Dos Santos, Thibaut Issenhuth, Emmanuel de Bézenac, Mickaël Chen, Alain Rakotomamonjy

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import enum


@enum.unique
class Test(str, enum.Enum):
    # Available tests
    FID = 'fid'  # FID
    GEN_SAVE = 'gen_save'  # Saves generated images in a Tensor file
    GEN_QUALI = 'gen_quali'  # Saves some generated images in a (A)PNG file.


quanti_test = [Test.FID]
