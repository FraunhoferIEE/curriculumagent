"""
In this file, a PPO model is reproduced following OPENAI baselines' PPO model.
@https://github.com/openai/baselines/tree/master/baselines/ppo2

The model is the orignal approach of @https://github.com/AsprinChina/L2RPN_NIPS_2020_a_PPO_Solution
"""
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf


class PPO(tf.Module):
    def __init__(self, coef_entropy: float = 0.01, coef_value_func: float = 0.5, max_grad_norm: float = 0.5):
        """ PPO Class for the Senior model.
        Within the constructor, the hyper-parameters of the PPO are set.
        The PPO Model is based on the OpenAI Baseline PPO2

        Args:
            coef_entropy: entropy of PPO
            coef_value_func: parameter of the value function
            max_grad_norm: The maximum value for the gradient clipping
        """
        super(PPO, self).__init__()
        self.model = Policy_Value_Network()
        self.optimizer = tf.keras.optimizers.Adam()
        self.coef_entropy = coef_entropy
        self.coef_value_func = coef_value_func
        self.max_grad_norm = max_grad_norm
        self.step = self.model.step
        self.value = self.model.value
        self.initial_state = None
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approx_kl', 'clip_ratio']

    def train(self, obs: tf.Tensor, returns: tf.Tensor, actions: tf.Tensor, values: tf.Tensor,
              neg_log_p_old: tf.Tensor, advs: tf.Tensor, lr: float = 3e-4, clip_range: float = 0.2) -> List:
        """ Train method for the PPO method
        Args:
            obs: aggregated observations of multiple runs
            returns: return values of multiple runs, i.e., comparable to reward
            actions: aggregated observations of multiple runs
            values: aggregated value function of multiple runs
            neg_log_p_old: old negative log likelihood
            advs: difference between the forecasted return (value) and the true return
            lr: learning rate
            clip_range: clipping parameter

        Returns:

        """
        grads, policy_loss, value_loss, entropy, approx_kl, clip_ratio = self._get_grad(obs, returns, actions, values,
                                                                                        neg_log_p_old, advs, clip_range)
        self.optimizer.learning_rate = lr
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return policy_loss, value_loss, entropy, approx_kl, clip_ratio

    @tf.function
    def _get_grad(self, obs: tf.Tensor, returns, actions, values, neg_log_p_old, advs, clip_range) -> List:
        """ Method to receive the current gradient of the model based on the observation.

        Args:
            obs: aggregated observations of multiple runs
            returns: return values of multiple runs, i.e., comparable to reward
            actions: aggregated observations of multiple runs
            values: aggregated value function of multiple runs
            neg_log_p_old: old negative log likelihood
            advs: difference between the forecasted return (value) and the true return
            clip_range: clipping parameter

        Returns:

        """
        # advs = returns - values
        # advs = (advs - tf.reduce_mean(advs)) / (tf.keras.backend.std(advs) + 1e-8)

        # obtain current gradient
        with tf.GradientTape() as tape:
            logit, _ , _ = self.model.model(obs)
            actions = tf.cast(actions, tf.int32)
            actions_one_hot = tf.one_hot(actions, logit.get_shape().as_list()[-1])
            neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=actions_one_hot)

            # calculate entropy bonus
            entropy = tf.reduce_mean(self._get_entropy(logit))

            # calculate value loss
            vpred = self.model.value(obs)
            vpred_clip = values + tf.clip_by_value(vpred - values, -clip_range, clip_range)
            value_loss1 = tf.square(vpred - returns)
            value_loss2 = tf.square(vpred_clip - returns)
            value_loss = 0.5 * tf.reduce_mean(tf.maximum(value_loss1, value_loss2))

            # calculate policy loss
            ratio = tf.exp(neg_log_p_old - neg_log_p)
            policy_loss1 = -advs * ratio
            policy_loss2 = -advs * tf.clip_by_value(ratio, (1 - clip_range), (1 + clip_range))
            policy_loss = tf.reduce_mean(tf.maximum(policy_loss1, policy_loss2))

            approx_kl = 0.5 * tf.reduce_mean(tf.square(neg_log_p_old - neg_log_p))
            clip_ratio = tf.reduce_mean(tf.cast(tf.greater(tf.abs(ratio - 1), clip_range), tf.float32))

            # Sigma loss
            loss = policy_loss * 10 + value_loss * self.coef_value_func - entropy * self.coef_entropy

        var_list = self.model.trainable_variables
        grads = tape.gradient(loss, var_list)
        grads, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
        return grads, policy_loss * 10, value_loss * self.coef_value_func, entropy, approx_kl, clip_ratio

    @staticmethod
    def _get_entropy(logit: tf.Tensor) -> tf.Tensor:
        """ Calculate the entropy for a given logit distribution

        The entropy is used to assess, whether the probability between the actions were similar or
        whether some actions had a higher probability.

        Args:
            logit: logit values of the model, which was computed based on an observation

        Returns: Tensor, which depicts the entropy between the different logit values

        """
        a0 = logit - tf.reduce_max(logit, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)


class Policy_Value_Network(tf.Module):
    def __init__(self, junior_path: Path = './JuniorModel'):
        """ Policy Value Network for the PPO optimization. The model is initialized with PVNet class.
        Afterwards the weights of the Junior Model are used to jumpstart the Senior model.


        Args:
            junior_path: Path of the Junior model checkpoints.
        """
        super().__init__()
        self.model = PVNet()
        self.model.build((None, 1221,))
        self._params_copy(junior_path)

    def _params_copy(self, path: Path):
        """ Private Method.

        Used to copy the weights of the Junior model onto the model of the PVNet

        Args:
            path: Path of the Junior model checkpoints.

        Returns: None

        """
        model = tf.keras.models.load_model(path)
        self.model.layers[0].set_weights(model.layers[0].get_weights())
        self.model.layers[1].set_weights(model.layers[1].get_weights())
        self.model.layers[2].set_weights(model.layers[2].get_weights())
        self.model.layers[3].set_weights(model.layers[4].get_weights())
        self.model.layers[4].set_weights((*map(lambda x: x / 5, model.layers[6].get_weights()),))

    @tf.function
    def step(self, obs: tf.Tensor) -> List:
        """ Step method that executes the action step of the PVNet Model

        Args:
            obs: observation of Grid2Op in tf.keras.model readable format

        Returns: list containing action, state value, negative log propability and logit value

        """
        # l for logits, p for possibility, and v for value
        logit, _, state_value = self.model(obs)
        # sampling by Gumbel-max trick
        u = tf.random.uniform(tf.shape(logit), dtype=np.float32)
        a = tf.argmax(logit - tf.math.log(tf.math.negative(tf.math.log(u))), axis=-1)
        a_one_hot = tf.one_hot(a, logit.get_shape().as_list()[-1])  # important!
        # calculate -log(pi)
        neg_log_p = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=a_one_hot)
        state_value = tf.squeeze(state_value, axis=1)
        return a, state_value, neg_log_p, logit

    @tf.function
    def value(self, obs: tf.Tensor) -> tf.Tensor:
        """ Compute the value of an observation

        Args:
            obs: observation of Grid2Op in tf.keras.model readable format

        Returns: value of the observation

        """
        _, _, value = self.model(obs)
        value = tf.squeeze(value, axis=1)
        return value


class PVNet(tf.keras.Model):
    def __init__(self):
        """ Constructor of the PVNet Model.
        The model is similar to the Junior model, however two additional layers are added in order to
        compute the PPO.

        The number of cells is set to 1000!
        """
        super(PVNet, self).__init__()
        n_cell = 1000
        initializer = tf.keras.initializers.Orthogonal()
        self.layer1 = tf.keras.layers.Dense(n_cell, activation='relu', kernel_initializer=initializer)
        self.layer2 = tf.keras.layers.Dense(n_cell, activation='relu', kernel_initializer=initializer)
        self.layer3 = tf.keras.layers.Dense(n_cell, activation='relu', kernel_initializer=initializer)
        self.layer4 = tf.keras.layers.Dense(n_cell, activation='relu', kernel_initializer=initializer)
        self.act_layer = tf.keras.layers.Dense(208, activation=None, kernel_initializer=initializer)
        self.val_hidden_layer = tf.keras.layers.Dense(64, activation='relu', kernel_initializer=initializer)
        self.val_layer = tf.keras.layers.Dense(1, activation=None, kernel_initializer=initializer)

    def call(self, input: tf.Tensor):
        """ Call method of the model.

        In the call method, the input is forward passed and the logits, probabilities and
        the state values are returned.

        Args:
            input: input of the model, which is forward passed.

        Returns: Returns the logits, the probability distribution as well as the state value of the model.

        """
        out1 = self.layer1(input)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        logits = self.act_layer(out4)  # logits
        prob = tf.nn.softmax(l)  # probability distribution of actions
        vh = self.val_hidden_layer(out4)
        value_of_state = self.val_layer(vh)  # state value
        return logits, prob, value_of_state


if __name__ == '__main__':
    # for test only
    m = Policy_Value_Network()
    l, p, v = m.model(np.ones((1, 1221)))
