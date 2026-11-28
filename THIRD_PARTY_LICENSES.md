# Third-party code and licences

Parts of this repository are copied or adapted from other projects; they keep their
licences. The rest of the repository is under CC BY-NC 4.0 ([LICENSE](LICENSE)).

| Files | Origin | Licence |
|---|---|---|
| `ldpt/models/dinov3/` | [DINOv3](https://github.com/facebookresearch/dinov3), Meta Platforms, Inc. and affiliates (inference copy of the backbone) | DINOv3 License Agreement (`LICENSE.md` in the DINOv3 repository) |
| `ldpt/utils/comm.py`, `ldpt/utils/timer.py`, `ldpt/utils/events.py`, `ldpt/engines/default.py`, `ldpt/engines/launch.py` | [detectron2](https://github.com/facebookresearch/detectron2), Facebook, Inc. and its affiliates (modified) | Apache License 2.0 |
| `ldpt/utils/config.py`, `ldpt/utils/registry.py`, `ldpt/utils/path.py`, `ldpt/utils/logger.py` | [MMCV](https://github.com/open-mmlab/mmcv), OpenMMLab (modified) | Apache License 2.0 |
| `ldpt/models/utils/depth.py` (`_roe_align`, `_roe_support`, the ROE solver) | [MoGe](https://github.com/microsoft/MoGe), `moge/utils/alignment.py` at commit 74fbce0, Microsoft Corporation (ported) | MIT License |
| `ldpt/datasets/transforms.py` (edge-aware depth resampling) | the rule of MoGe's data loader and utils3d's `depth_map_edge` (reimplemented) | MIT License |
| `ldpt/models/losses/lovasz.py` | [Lovasz-Softmax](https://github.com/bermanmaxim/LovaszSoftmax) (Berman et al., CVPR 2018), Maxim Berman, via [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt) | MIT License |
| trainer, hooks, registries and config layout (`ldpt/engines/`, `ldpt/hooks/`, `train.py`, `train.sh`) | [Pointcept](https://github.com/Pointcept/Pointcept) | MIT License |

Model weights are not part of this repository. The DINOv3 weights are Meta's, under the
DINOv3 License, and are not redistributed. The L-DPT decoder weights are released on
[Hugging Face](https://huggingface.co/pointcept-bot/L-DPT) under CC BY-NC 4.0; the
training datasets keep their own terms
([DepthHybrid](https://huggingface.co/datasets/pointcept-bot/DepthHybrid) lists them).

## MIT License (MoGe, Pointcept, pytorch-toolbelt, Lovasz-Softmax)

The copyright notices of the four MIT-licensed sources, MoGe, Pointcept,
pytorch-toolbelt and Lovasz-Softmax, in this order, and their licence text:

```
MIT License

Copyright (c) Microsoft Corporation.
Copyright (c) 2023 Pointcept
Copyright (c) Eugene Khvedchenya
Copyright (c) 2018 Maxim Berman

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The Apache License 2.0 texts are at <https://www.apache.org/licenses/LICENSE-2.0>; the
DINOv3 License Agreement is at <https://github.com/facebookresearch/dinov3>.
