<!-- badges: start -->

![Quarto Website Publish](https://github.com/ojwatson/emidm.io/actions/workflows/publish.yml/badge.svg)
[![Website](https://img.shields.io/badge/View-Website-blue?logo=githubpages&style=flat-square)](https://ojwatson.github.io/emidm/)

# emidm


## Table of Contents

- [Installation](#installation)
- [Example Usage](#example-usage)
- [License](#license)

## Installation

``` bash
pip install git+https://github.com/OJWatson/emidm.git
```

Or

``` bash
git clone https://github.com/OJWatson/emidm.git
cd emidm
pip install .
```

## Example Usage

``` python
from emidm.sir import run_sir, run_model_with_replicates, plot_model_outputs
single = run_sir()
single.plot("t", ["S", "I", "R"])
```

![](_freeze/index/figure-html/cell-2-output-1.png)

## License

`emidm` is distributed under the terms of the
[MIT](https://spdx.org/licenses/MIT.html) license.
