#!/usr/bin/env python

from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tqdm import tqdm

import unittest


def batch_apply_tf(f, batch_size, *args,
                   function=False, progress=False, numpy=False):
    """
    Applies a tensorflow function f(*args), but internally batches the args,
    and then concatenates the results. The internal batching is usually
    meant to avoid running out of memory.

    Inputs:
      f (function): The function to apply.
      batch_size (int): Batch size.
      *args (array-like): The arguments to pass to `f`. Each argument will be
             batched along its zeroeth axis.
      function (Optional[bool]): If `True`, the function `f` will be
                                 wrapped using tf.function. Defaults to
                                 `False`.
      progress (Optional[bool]): If `True`, a progress bar will be shown.
                                 Defaults to `False`.
      numpy (Optional[bool]): If `True`, outputs will be converted to numpy
                              arrays.

    Outputs:
      res: Should be the same as the output of f(*args). The batching should
           have no impact on the final result.
    """
    res = []
    def f_batch(*x):
        return f(*x)
    if function:
        f_batch = tf.function(f_batch)
    iterator = range(0, len(args[0]), batch_size)
    if progress:
        iterator = tqdm(iterator)
    for i in iterator:
        batch = [a[i:i+batch_size] for a in args]
        res_batch = f_batch(*batch)
        if numpy:
            res_batch = res_batch.numpy()
        res.append(res_batch)
    if numpy:
        res = np.concatenate(res, axis=0)
    else:
        res = tf.concat(res, axis=0)
    return res


class TestBatchApplyTF(unittest.TestCase):
    def test_apply(self):
        """
        Tests that batch_apply_tf(f, batch_size, *args) produces identical
        results to f(*args), with various batch sizes and optional arguments.
        """
        n = 1024
        x = tf.random.uniform([n])
        y = tf.random.uniform([n,3])
        funcs = [
            lambda xx,yy: tf.math.cos(xx) * tf.math.reduce_sum(yy,axis=1),
            lambda xx,yy: tf.expand_dims(tf.math.cos(xx),1) * yy
        ]
        for f in funcs:
            res_direct = f(x, y)
            for batch_size in [4, 16, 32, 64, 1024, 5, 21]:
                # Batching, with no further options
                res_batch = batch_apply_tf(f, batch_size, x, y)
                self.assertEqual(res_batch.shape, res_direct.shape)
                np.testing.assert_allclose(res_batch, res_direct,
                                           atol=1e-5, rtol=1e-5)
                # Batching, with tf.function wrapper
                res_batch = batch_apply_tf(f, batch_size, x, y, function=True)
                self.assertEqual(res_batch.shape, res_direct.shape)
                np.testing.assert_allclose(res_batch, res_direct,
                                           atol=1e-5, rtol=1e-5)
                # Batching, with numpy conversion
                res_batch = batch_apply_tf(f, batch_size, x, y, numpy=True)
                self.assertEqual(res_batch.shape, res_direct.shape)
                np.testing.assert_allclose(res_batch, res_direct,
                                           atol=1e-5, rtol=1e-5)
                self.assertIsInstance(res_batch, np.ndarray)
                # Batching, with progressbar
                res_batch = batch_apply_tf(f, batch_size, x, y, progress=True)
                self.assertEqual(res_batch.shape, res_direct.shape)
                np.testing.assert_allclose(res_batch, res_direct,
                                           atol=1e-5, rtol=1e-5)


def main():
    unittest.main()

    return 0

if __name__ == '__main__':
    main()

