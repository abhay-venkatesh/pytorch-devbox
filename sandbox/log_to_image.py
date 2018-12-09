import matplotlib.pyplot as plt
import sys
import tensorflow as tf

if len(sys.argv) < 2:
    print("Usage: python main.py <path to log file>")
    exit()
logfile_path = sys.argv[1]

losses = []
for e in tf.train.summary_iterator(logfile_path):
    for v in e.summary.value:
        if v.tag == 'loss':
            losses.append(v.simple_value)

plt.plot(losses)
plt.ylabel('Loss')
plt.savefig('loss.png')