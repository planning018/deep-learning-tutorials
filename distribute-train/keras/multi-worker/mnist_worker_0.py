import os
import json

import tensorflow as tf
import mnist_setup

def set_tf_config(type, index):
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "chief": ["127.0.0.1:12345"],  # 调度节点
            "worker": ["127.0.0.1:23456", "127.0.0.1:23457"]  # 计算节点
        },
        "task": {"type": type, "index": index}  # 定义本进程为worker节点，即["127.0.0.1:5001"]为计算节点
    })


# mock TF_CONFIG environment variable
type = 'worker'
set_tf_config(type, 0)

per_worker_batch_size = 64
tf_config = json.loads(os.environ['TF_CONFIG'])
num_workers = len(tf_config['cluster']['worker'])

strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_setup.mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = mnist_setup.build_and_compile_cnn_model()


multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)