import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import os

if len(sys.argv) < 2:
    print("Usage: python main.py <path to log file folder>")
    exit()
logdir = sys.argv[1]

losses = []
directory = os.fsencode(logdir)
for logfile in os.listdir(directory):
    filename = os.fsdecode(logfile)
    file_path = os.path.join(logdir, filename)
    for e in tf.train.summary_iterator(file_path):
        for v in e.summary.value:
            if v.tag == 'loss':
                losses.append(v.simple_value)

plt.plot(losses)
plt.ylabel('Loss')
plt.savefig('loss.png')