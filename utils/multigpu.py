
import tensorflow as tf
import pdb

def average_gradients(tower_grads):

    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)

        grad = tf.concat(grads, axis=0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_vars = (grad,v)
        average_grads.append(grad_and_vars)

    return average_grads
