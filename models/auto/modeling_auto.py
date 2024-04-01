from typing import OrderedDict
from transformers.configuration_utils import PretrainedConfig
from models.auto.configuration_auto import AutoArchConfig

from models.t5.configuration_t5 import T5Config
from models.t5.modeling_t5 import T5ForConditionalGeneration

from models.bart.configuration_bart import BartConfig
from models.bart.modeling_bart import BartForConditionalGeneration

MODEL_FOR_CONDITIONAL_GENERATION_MAPPING = OrderedDict(
    [
        (T5Config, T5ForConditionalGeneration),
        (BartConfig, BartForConditionalGeneration)
    ]
)

class AutoArchForSeq2SeqLM:
    r"""
    This is a generic model class that will be instantiated as one of the model classes of the library---with a
    sequence-to-sequence language modeling head---when created with the
    :meth:`~transformers.AutoModelForSeq2SeqLM.from_pretrained` class method or the
    :meth:`~transformers.AutoModelForSeq2SeqLM.from_config` class method.

    This class cannot be instantiated directly using ``__init__()`` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoArchForSeq2SeqLM is designed to be instantiated "
            "using the `AutoArchForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoArchForSeq2SeqLM.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        r"""
        Instantiates one of the model classes of the library---with a sequence-to-sequence language modeling
        head---from a configuration.

        Note:
            Loading a model from its configuration file does **not** load the model weights. It only affects the
            model's configuration. Use :meth:`~transformers.AutoModelForSeq2SeqLM.from_pretrained` to load the model
            weights.

        Args:
            config (:class:`~transformers.PretrainedConfig`):
                The model class to instantiate is selected based on the configuration class:

                List options

        Examples::

            >>> from transformers import AutoArchConfig, AutoModelForSeq2SeqLM
            >>> # Download configuration from huggingface.co and cache.
            >>> config = AutoArchConfig.from_pretrained('t5')
            >>> model = AutoModelForSeq2SeqLM.from_config(config)
        """
        if type(config) in MODEL_FOR_CONDITIONAL_GENERATION_MAPPING.keys():
            return MODEL_FOR_CONDITIONAL_GENERATION_MAPPING[type(config)](config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_CONDITIONAL_GENERATION_MAPPING.keys()),
            )
        )


    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Examples::

            >>> from transformers import AutoArchConfig, AutoModelForSeq2SeqLM

            >>> # Download model and configuration from huggingface.co and cache.
            >>> model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')

            >>> # Update configuration during loading
            >>> model = AutoModelForSeq2SeqLM.from_pretrained('t5-base', output_attentions=True)
            >>> model.config.output_attentions
            True

            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AutoArchConfig.from_json_file('./tf_model/t5_tf_model_config.json')
            >>> model = AutoModelForSeq2SeqLM.from_pretrained('./tf_model/t5_tf_checkpoint.ckpt.index', from_tf=True, config=config)
        """
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config, kwargs = AutoArchConfig.from_pretrained(
                pretrained_model_name_or_path, return_unused_kwargs=True, **kwargs
            )

        if type(config) in MODEL_FOR_CONDITIONAL_GENERATION_MAPPING.keys():
            return MODEL_FOR_CONDITIONAL_GENERATION_MAPPING[type(config)].from_pretrained(
                pretrained_model_name_or_path, *model_args, config=config, **kwargs
            )
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_CONDITIONAL_GENERATION_MAPPING.keys()),
            )
        )
