
import tensorflow as tf
TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(list("ACGTNacgtn")),
        values=tf.constant([0,1,2,3,4,0,1,2,3,4], dtype=tf.int32)
    ), default_value=4
)
def one_hot_dna(seq:str)->tf.Tensor:
    x=tf.constant(list(seq))
    idx=TABLE.lookup(x)
    return tf.one_hot(idx, depth=5, dtype=tf.float32)
def revcomp_one_hot(oh: tf.Tensor)->tf.Tensor:
    perm=tf.constant([3,2,1,0,4])
    rc=tf.gather(oh, perm, axis=-1)
    rc=tf.reverse(rc, axis=[0])
    return rc
