# Fish Audio S2-Pro — Research License Notice

This project depends on the Fish Audio S2-Pro model weights at runtime. The
weights are NOT distributed with this repository; operators bind-mount them
into the dubbing-generator container at deploy time.

The weights are licensed under the Fish Audio Research License, Copyright
© 39 AI, INC. All Rights Reserved.

- Research and non-commercial use: free, under the terms of the upstream
  Agreement.
- Commercial use: requires a separate written license from Fish Audio
  (business@fish.audio).
- Derivative outputs (dubbed videos generated using these weights) inherit
  the non-commercial restriction unless a commercial license is in place.

The s2.cpp inference engine source code is itself a Derivative Work of the
Fish Audio Materials and is distributed under the same Research License
terms.

For full license text, see https://huggingface.co/fishaudio/s2-pro and
https://github.com/rodrigomatta/s2.cpp/blob/main/LICENSE.md.
