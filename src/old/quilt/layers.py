
import tensorflow as tf
from .tokenize import revcomp_one_hot
class RevComp(tf.keras.layers.Layer):
    def call(self, x): return revcomp_one_hot(x)
class RCInvariant(tf.keras.layers.Layer):
    def __init__(self, base, reduce="mean", **kw):
        super().__init__(**kw); self.base=base; self.reduce=reduce
    def call(self, x, training=False):
        y1=self.base(x, training=training)
        y2=self.base(revcomp_one_hot(x), training=training)
        y = tf.maximum(y1,y2) if self.reduce=="max" else (y1+y2)/2.0
        return tf.math.l2_normalize(y, axis=-1)
