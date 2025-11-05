
import tensorflow as tf
class DotProductIndex:
    def __init__(self, vectors: tf.Tensor, ids):
        self.vectors=tf.math.l2_normalize(tf.convert_to_tensor(vectors, tf.float32), axis=-1)
        self.ids=list(ids)
    @classmethod
    def from_numpy(cls, np_vectors, ids):
        return cls(tf.convert_to_tensor(np_vectors, tf.float32), ids)
    def search(self, queries: tf.Tensor, k=5):
        q=tf.math.l2_normalize(queries, axis=-1)
        sims=tf.linalg.matmul(q, self.vectors, transpose_b=True)
        vals, idx=tf.math.top_k(sims, k=k, sorted=True)
        sel_ids=tf.convert_to_tensor([self.ids[i] for i in idx.numpy().ravel()], dtype=tf.string)
        sel_ids=tf.reshape(sel_ids, idx.shape)
        return vals.numpy(), sel_ids.numpy()
