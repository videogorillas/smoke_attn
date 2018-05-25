#!/usr/bin/env python
# coding: utf-8

import os
import random
import sys

f = sys.argv[1]
print(f)
with open(f, 'r') as _f:
    n = _f.readlines()
random.shuffle(n)

split_index = int(len(n) * 0.1)

bfn = os.path.basename(f)
fn = "validate_" + bfn
with open(fn, 'w') as ntest:
    for l in n[:split_index]:
        ntest.write(l)
print(fn)

fn = "train_" + bfn
with open(fn, 'w') as ntrain:
    for l in n[split_index:]:
        ntrain.write(l)

print(fn)
