import tensorflow as tf

class MyModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    https://github.com/tensorflow/tensorflow/issues/33163
    """

    def __init__(self, *args, epoch_per_save=1, **kwargs):
        self.epochs_per_save = epoch_per_save
        super().__init__(*args, save_freq="epoch", **kwargs)

    def on_epoch_end(self, epoch, logs):
        if epoch % self.epochs_per_save == 0:
            super().on_epoch_end(epoch, logs)

    def best_val_loss(self):
        return self.best