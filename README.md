# JRD_EXTENSIONS

![Python Version](https://img.shields.io/badge/Python->=3.10-blue)
![Code Style](https://img.shields.io/badge/Code_Style-black-black)

[**Features**](#features)
| [**Installation**](#installation) 


## Features

<details> 
<summary> PRNGSequence </summary>

- Without extension :
```python
import jax
key = jax.random.key(0)
for _ in iterations:
    key, _k = jax.random.split(key)
    do_random_things(_k)
```

- With extension : 
```python
import jrd_extensions
key = jrd_extensions.PRNGSequence(0)
for _ in iterations:
    do_random_things(next(k))
```
</details>

## Installation
This package requires Python 3.10 or later and a working [JAX](https://github.com/google/jax) installation.
To install JAX, refer to [the instructions](https://github.com/google/jax#installation).

```bash
pip install --upgrade pip
pip install git+https://github.com/Raffaelbdl/jrd_extensions
```