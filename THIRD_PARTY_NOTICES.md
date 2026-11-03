# Attribution and distribution notes

This repository contains the RTAT implementation, portable research utilities and the derived external/calendar inputs described in external_data/ATTRIBUTION.md. It does not assign an open-source license on behalf of the authors. A public-release license can be added by the rights holders when the repository is published. Public hosting does not change third-party data terms.

The processed benchmark is attributed to:

Xiao Yan, Xianghua Gan, Jingjing Tang, Dapeng Zhang, and Rui Wang (2024). *ProSTformer: Progressive Space-Time Self-Attention Model for Short-Term Traffic Flow Forecasting*. IEEE Transactions on Intelligent Transportation Systems, 25(9), 10802–10816. DOI: 10.1109/TITS.2024.3367754.

Repository: https://github.com/yanxiao1930/ProSTformer_code

The underlying trip records are provided by NYC TLC and Citi Bike. Weather inputs are ERA5 reanalysis obtained through Open-Meteo. Their respective attribution and usage terms apply. The repository does not redistribute the processed flow datasets or raw weather responses. Derived weather features and reconstructed calendar metadata are included with their source attribution and upstream data licence links in external_data/ATTRIBUTION.md.

PyTorch, NumPy and the IANA time-zone data are external dependencies with their own licenses. The model uses hierarchical local-window and shifted-window concepts; methodological references are given in the accompanying manuscript. Baseline third-party repositories are not copied into this release.
