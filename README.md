# emidm

[![PyPI - Version](https://img.shields.io/pypi/v/emidm.svg)](https://pypi.org/project/emidm)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/emidm.svg)](https://pypi.org/project/emidm)

-----

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
git clone https://github.com/OJwatson/emidm.git
cd emidm
pip install .
```

## Example Usage

``` python 
>>> from emidm.sir import run_sir, run_model_with_replicates, plot_model_outputs
>>> single = run_sir()
>>> single
	t	N	S	I	R
0	0	1000	990	10	0
1	1	1000	984	16	0
2	2	1000	981	16	3
3	3	1000	977	16	7
4	4	1000	974	18	8
...	...	...	...	...	...
96	96	1000	223	16	761
97	97	1000	221	14	765
98	98	1000	220	14	766
99	99	1000	220	14	766
100	100	1000	220	12	768
```

## License

`emidm` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
