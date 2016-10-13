A video and caption matching system
=======

# Introduction

# Prerequisites

* Python 2.7.x or 3.5.x

* Keras installation with Theano or Tensorflow backend, but *with tensorflow matrix order*. If you are not sure how to setup, use the following configuration in your `~/.keras/keras.json`:

```
{
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
```

	* A configuration for Theano (use in `~/.theanorc`, with GPU+CUDNN):

```
[global]
floatX = float32 # from cuda-8.0 can we use float16?
device = gpu
optimizer = fast_compile
mode = FAST_RUN

[dnn]
enabled = True

[lib]
cnmem = 1 # maybe it is good
```

# Structure

# Contact and future maintain

I actually do not have a plan to maintain this Python implement in future (I'm moving all my codebase to Java now, sadly). For any technical discussions, please file an [issue](https://github.com/marker68/video-matching/issues/new).
