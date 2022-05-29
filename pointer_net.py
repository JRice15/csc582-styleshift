import numpy as np
import tensorflow as tf


# class ScalarDense()


class PointerNet(tf.keras.layers.Layer):

    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size

        self.p_gen_dense = tf.keras.layers.Dense(1)
    
    def call(self, inp_tokens, tar_embedded, generator_output, enc_output, 
            dec_state, attn_heads):
        """
        Following https://aclanthology.org/2020.aacl-srw.13.pdf

        shapes:
        B = batchsize
        T = target seq len
        I = input seq len
        H = num heads
        V = vocab size
        D = d_model

        inp_tokens: (B, I)
        tar_embedded: (B, T, D)
        generator_output: (B, T, V)
        enc_output: (B, I, D)
        dec_state: (B, T, D)
        attn_heads: (B, H, T, I)

        output shape: (B, T, V)
        """
        # average over heads
        attn = tf.reduce_mean(attn_heads, axis=1) # (B, T, I)

        ### Probability of generating vs pointing, P_gen
        context = tf.einsum("bti,bid->btd", attn, enc_output)
        p_gen_inputs = tf.concat([context, dec_state, tar_embedded], axis=-1) # (B, T, D*3)

        p_gen = self.p_gen_dense(p_gen_inputs) # (B, T, 1)
        p_gen = tf.math.sigmoid(p_gen) # (B, T, 1)

        # regularize against p_gen values <0.05 or >0.95
        # this loss maxes out at 0.5, for p_gen == 1 or 0. The constant factor at the start controls the steepness of the loss
        p_gen_loss = 10 * tf.nn.relu(tf.abs(p_gen - 0.5) - 0.45)
        p_gen_loss = tf.reduce_mean(p_gen_loss)
        self.add_loss(p_gen_loss)
        self.add_metric(p_gen_loss, name="pgen_reg_loss")
        self.add_metric(tf.reduce_mean(p_gen), name="pgen_avg")

        ### Pointer output
        inp = tf.one_hot(inp_tokens, depth=self.vocab_size) # (B, I, V)
        pointer_output = tf.einsum("bti,biv->btv", attn, inp) # (B, T, V)
        pointer_output = tf.math.softmax(pointer_output, axis=-1) # (B, T, V)

        ### Final outputs 
        final_output = (p_gen * generator_output) + ((1 - p_gen) * pointer_output) # (B, T, V)

        pointer_data = {
            "pointer_distribution": pointer_output,
            "p_gen": tf.squeeze(p_gen, axis=-1)
        }
        return final_output, pointer_data

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            **super().get_config(),
        }