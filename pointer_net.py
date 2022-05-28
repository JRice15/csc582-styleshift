import numpy as np
import tensorflow as tf
import tensorflow_probability


# class ScalarDense()


class PointerNet(tf.keras.layers.Layer):

    def __init__(self, vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size

        self.p_gen_dense = tf.keras.layers.Dense(1)

        self.pointer_scale_w = tf.Variable(
            initial_value=1.0,
            name="pointer_scale_weight",
            trainable=True,
        )
        self.pointer_scale_b = tf.Variable(
            initial_value=0.0,
            name="pointer_scale_bias",
            trainable=True,
        )
    
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

        ### P_gen
        context = tf.einsum("bti,bid->btd", attn, enc_output)
        p_gen_inputs = tf.concat([context, dec_state, tar_embedded], axis=-1) # (B, T, D*3)

        p_gen = self.p_gen_dense(p_gen_inputs) # (B, T, 1)
        p_gen = tf.math.sigmoid(p_gen) # (B, T, 1)

        ### Pointer output
        inp = tf.one_hot(inp_tokens, depth=self.vocab_size) # (B, I, V)
        pointer_output = tf.einsum("bti,biv->btv", attn, inp)
        # scale the [0,1] constrained pointer_output into some logit space that hopefully is approximates the space of generator_output
        pointer_output = tf.nn.log_softmax(pointer_output, axis=-1)
        pointer_output = (self.pointer_scale_w * pointer_output) + self.pointer_scale_b

        ### Final outputs (summing in logit space)
        final_output = (p_gen * generator_output) + ((1 - p_gen) * pointer_output)

        pointer_data = {
            "pointer_logits": pointer_output,
            "p_gen": tf.squeeze(p_gen, axis=-1)
        }
        return final_output, pointer_data
        

    def get_config(self):
        return {
            "vocab_size": self.vocab_size,
            **super().get_config(),
        }