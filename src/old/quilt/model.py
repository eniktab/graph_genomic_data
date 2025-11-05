
import tensorflow as tf
def conv_block(x, filters, kernel, dilation=1):
    x = tf.keras.layers.Conv1D(filters, kernel, padding="same", dilation_rate=dilation, activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("gelu")(x)
    x = tf.keras.layers.Conv1D(filters, 1, padding="same", activation=None, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.Activation("gelu")(x)
def build_read_embedder_base(seq_len=None, d_model=256, channels=5):
    inp=tf.keras.Input(shape=(seq_len, channels))
    x=tf.keras.layers.Conv1D(64,9,padding="same",activation="gelu")(inp)
    for dil in [1,2,4,8,16]:
        x=conv_block(x,128,7,dilation=dil)
        x=tf.keras.layers.AveragePooling1D(2,padding="same")(x)
    x=tf.keras.layers.Conv1D(256,3,padding="same",activation="gelu")(x)
    x=tf.keras.layers.GlobalAveragePooling1D()(x)
    x=tf.keras.layers.Dense(d_model,activation=None)(x)
    out=tf.math.l2_normalize(x,axis=-1)
    return tf.keras.Model(inp,out,name="read_embedder_base")
def build_read_embedder_rc(seq_len=None, d_model=256, channels=5, reduce="mean"):
    base=build_read_embedder_base(seq_len=seq_len, d_model=d_model, channels=channels)
    from .layers import RCInvariant
    inp=tf.keras.Input(shape=(seq_len, channels))
    emb=RCInvariant(base, reduce=reduce)(inp)
    return tf.keras.Model(inp, emb, name="read_embedder_rc")
