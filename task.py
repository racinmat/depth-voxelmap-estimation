# encoding: utf-8

from tensorflow.python.platform import gfile
import tensorflow as tf
import Network
import time

current_time = time.strftime("%Y-%m-%d--%H-%M-%S", time.gmtime())


def train():
    network = Network.Network()
    network.train()


def main(argv=None):
    if not gfile.Exists("./train"):
        gfile.MakeDirs("./train")
    if not gfile.Exists("./test"):
        gfile.MakeDirs("./test")
    if not gfile.Exists(Network.PREDICT_DIR):
        gfile.MakeDirs(Network.PREDICT_DIR)
    if not gfile.Exists(Network.CHECKPOINT_DIR):
        gfile.MakeDirs(Network.CHECKPOINT_DIR)
    if not gfile.Exists(Network.LOGS_DIR):
        gfile.MakeDirs(Network.LOGS_DIR)
    train()


if __name__ == '__main__':
    tf.app.run()
