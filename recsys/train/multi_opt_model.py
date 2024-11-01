from keras.src.mixed_precision import loss_scale_optimizer as lso
from keras.src.engine import data_adapter
from keras.src.saving import saving_lib

# isort: off
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.python.platform import tf_logging as logging


DEFAULT = "default"


class Model(tf.keras.Model):
    """
    You could use multiple optimizers through calling:
    `model.compile(optimizer={"dnn": "adam", "embedding": "adagrad", "default": "adam"})`
    then those layers with prefix "dnn" will use the adam optimizer, and adagrad for prefix "embedding".
    Also, you must have the default optimizer for legacy layers.

    It should be noted that `model.save()` will crash when you want to save the optimizer,
    you could use the `tf.train.Checkpoint`, see `examples.pepnet_multi_opt.py`
    """
    def _get_optimizer(self, optimizer):
        def _get_single_optimizer(opt):
            opt = optimizers.get(opt)
            if self.dtype_policy.name == "mixed_float16" and not isinstance(
                opt, lso.BaseLossScaleOptimizer
            ):
                # Loss scaling is necessary with mixed_float16 for models to
                # converge to the same accuracy as with float32.
                opt = lso.BaseLossScaleOptimizer(opt)
            return opt

        if isinstance(optimizer, dict):
            assert DEFAULT in optimizer, "you must set the default optimizer"

            self.special_layer_variables = {}
            for prefix, opt in optimizer.items():
                optimizer[prefix] = tf.nest.map_structure(_get_single_optimizer, opt)
                self.special_layer_variables[prefix] = []

            for var in self.trainable_variables:
                default = True
                for prefix in optimizer:
                    if prefix == DEFAULT:
                        continue
                    if var.name.startswith(prefix):
                        self.special_layer_variables[prefix].append(var)
                        default = False
                        break
                if default:
                    self.special_layer_variables[DEFAULT].append(var)
            return optimizer
        else:
            return tf.nest.map_structure(_get_single_optimizer, optimizer)

    def train_step(self, data):
        """The logic for one training step.

        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happens in fit](
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.

        This method should contain the mathematical logic for one step of
        training.  This typically includes the forward pass, loss calculation,
        backpropagation, and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function`
        and `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Args:
          data: A nested structure of `Tensor`s.

        Returns:
          A `dict` containing values that will be passed to
          `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
          values of the `Model`'s metrics are returned. Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        # Run forward pass.
        persistent = True if isinstance(self.optimizer, dict) else False
        with tf.GradientTape(persistent=persistent) as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(x, y, y_pred, sample_weight)
        self._validate_target_and_loss(y, loss)
        # Run backwards pass.
        if isinstance(self.optimizer, dict):
            for layer in self.optimizer:
                self.optimizer[layer].minimize(loss, self.special_layer_variables[layer], tape=tape)

            del tape
        else:
            self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        return self.compute_metrics(x, y, y_pred, sample_weight)

    def compile_from_config(self, config):
        """Compiles the model with the information given in config.

        This method uses the information in the config (optimizer, loss,
        metrics, etc.) to compile the model.

        Args:
            config: Dict containing information for compiling the model.
        """
        has_overridden_compile = self.__class__.compile != Model.compile
        if has_overridden_compile:
            logging.warning(
                "`compile()` was not called as part of model loading "
                "because the model's `compile()` method is custom. "
                "All subclassed Models that have `compile()` "
                "overridden should also override "
                "`get_compile_config()` and `compile_from_config(config)`. "
                "Alternatively, you can "
                "call `compile()` manually after loading."
            )
            return
        config = saving_lib.deserialize_keras_object(config)
        self.compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            if isinstance(self.optimizer, dict):
                for layer in self.optimizer:
                    self.optimizer[layer].build(self.special_layer_variables[layer])
            else:
                self.optimizer.build(self.trainable_variables)