---
layout: post
description: Enable Python support in Neovim.
categories: [vim, conda, python]
title: "How to Setup Neovim with Python Provider using Conda"
comments: false
---

In this post I will go over the steps to setup Neovim with Python provider which is the Python language tool for Neovim. It is required by many Python specific plugins. I'll be using Conda environments to do this.

Firstly, if you don't have Neovim installed then install it. On Ubuntu type:

```sh
sudo apt install neovim
```

Neovim requires the pynvim Python package to enable Python support. It can be downloaded from PyPI using pip. If you use Python environments you will have to install pynvim for every environment in which you want to use Neovim. Fortunately, there is a work around, we can create a Python environment and install pynvim in it. Then in our init.vim we will tell Neovim to look for the Python provider in that environment.

I'll use Conda to create the environments. Of course you'll need Conda to be installed on your system. To create a Conda environment type:

```sh
conda create -n pynvim python=3.7
```

I named the environment pynvim and installed Python 3.7 in it. Now activate the environment by typing:

```sh
conda activate pynvim
```

Lets install pynvim. Type:

```sh
pip install pynvim
```

pynvim is installed in pynvim conda environment. We need to know the location of the environment. To do that type:

```sh
which python
```

Note the returned path we will need it. Now open your init.vim and add this line in it:

```vim
let g:python3_host_prog='/path/to/conda/environment'
```

Put the result of `which python` inside the single quotes `''`.

This is it. You've now configured Neovim for Python. You can now go ahead and install plugins. One really cool plugin is [Semshi](https://github.com/numirias/semshi) it does semantic syntax highlighting for Python.

I hope this was helpful for you.
