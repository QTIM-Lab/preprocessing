# preprocessing

A python library and CLI tool designed for the purpose of preprocessing MRI data for Deep Learning at QTIM. The `preprocessing` library revolves around the generation and continual updates of CSV files as code is executed, maintaining a complete overview of all filepaths and relevant metadata for an entire dataset as it changes over time.

## Tutorials

Please refer to the tutorials created in the notebooks directory within this repository. These tutorials guide the user through installation of the library, dataset generation, and finally preprocessing the data with several supported pipeline options. 

## Citations
`preprocessing` relies on the use of external tools for skullstripping and registration.

For skullstripping, we use [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/), which features in the following publications (listed chronologically):
> [SynthStrip: skull-stripping for any brain image](https://www.sciencedirect.com/science/article/pii/S1053811922005900)
> A. Hoopes, J.S. Mora, A.V. Dalca, B. Fischl*, and M. Hoffmann* (*equal contribution)

> [Boosting Skull-Stripping Performance for Pediatric Brain Images](https://arxiv.org/abs/2402.16634)
> W. Kelley, N. Ngo, A.V. Dalca, B. Fischl, L. Zöllei*, and M. Hoffmann* (*equal contribution)

[SynthMorph](https://martinos.org/malte/synthmorph/) is used for registration and features in the following publications (listed chronologically):

> [Learning MRI contrast-agnostic registration](https://martinos.org/malte/synthmorph/papers/hoffmann2021learning.pdf)
> M. Hoffmann, B. Billot, J.E. Iglesias, B. Fischl, and A.V. Dalca 

> [SynthMorph: learning contrast-invariant registration without acquired images](https://arxiv.org/abs/2004.10282)
> M. Hoffmann, B. Billot, D.N. Greve, J.E. Iglesias, B. Fischl, and A.V. Dalca

> [Anatomy-specific acquisition-agnostic affine registration learned from fictitious images](https://martinos.org/malte/synthmorph/papers/hoffmann2023anatomy.pdf)
> M. Hoffmann, A. Hoopes, B. Fischl*, A.V. Dalca* (*equal contribution)

> [Anatomy-aware and acquisition-agnostic joint registration with SynthMorph](https://arxiv.org/abs/2301.11329)
> M. Hoffmann, A. Hoopes, D.N. Greve, B. Fischl*, and A.V. Dalca* (*equal contribution) 
