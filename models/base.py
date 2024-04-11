from typing import Literal, Any
import typing_extensions
from pydantic import BaseModel


IncEx: typing_extensions.TypeAlias = 'set[int] | set[str] | dict[int, Any] | dict[str, Any] | None'

def try_load_model(model: BaseModel, val):
    try:
        if isinstance(val, model):
            return model.model_validate(val)
        elif isinstance(val, str):
            return model.model_validate_json(val)
        elif isinstance(val, dict):
            return model.model_validate(val)
    except Exception as e:
        return val

class CustomBaseModel(BaseModel):
    """Usage docs: https://docs.pydantic.dev/2.5/concepts/models/

    A base class for creating Pydantic models.

    Attributes:
        __class_vars__: The names of classvars defined on the model.
        __private_attributes__: Metadata about the private attributes of the model.
        __signature__: The signature for instantiating the model.

        __pydantic_complete__: Whether model building is completed, or if there are still undefined fields.
        __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
        __pydantic_custom_init__: Whether the model has a custom `__init__` function.
        __pydantic_decorators__: Metadata containing the decorators defined on the model.
            This replaces `Model.__validators__` and `Model.__root_validators__` from Pydantic V1.
        __pydantic_generic_metadata__: Metadata for generic models; contains data used for a similar purpose to
            __args__, __origin__, __parameters__ in typing-module generics. May eventually be replaced by these.
        __pydantic_parent_namespace__: Parent namespace of the model, used for automatic rebuilding of models.
        __pydantic_post_init__: The name of the post-init method for the model, if defined.
        __pydantic_root_model__: Whether the model is a `RootModel`.
        __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the model.
        __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the model.

        __pydantic_extra__: An instance attribute with the values of extra fields from validation when
            `model_config['extra'] == 'allow'`.
        __pydantic_fields_set__: An instance attribute with the names of fields explicitly set.
        __pydantic_private__: Instance attribute with the values of private attributes set on the model instance.
    """
    def model_dump(
        self,
        *,
        mode: Literal['json', 'python'] = 'python',
        include: IncEx = None,
        exclude: IncEx = None,
        by_alias: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool = True,
    ):
        """Usage docs: https://docs.pydantic.dev/2.5/concepts/serialization/#modelmodel_dump

        Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

        Args:
            mode: The mode in which `to_python` should run.
                If mode is 'json', the dictionary will only contain JSON serializable types.
                If mode is 'python', the dictionary may contain any Python objects.
            include: A list of fields to include in the output.
            exclude: A list of fields to exclude from the output.
            by_alias: Whether to use the field's alias in the dictionary key if defined.
            exclude_unset: Whether to exclude fields that have not been explicitly set.
            exclude_defaults: Whether to exclude fields that are set to their default value from the output.
            exclude_none: Whether to exclude fields that have a value of `None` from the output.
            round_trip: Whether to enable serialization and deserialization round-trip support.
            warnings: Whether to log warnings when invalid fields are encountered.

        Returns:
            A dictionary representation of the model.
        """
        return super().model_dump(
            mode=mode,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings
        )
    
    def model_dump_json(
        self,
        *,
        indent: int | None = None,
        include: IncEx = None,
        exclude: IncEx = None,
        by_alias: bool = True,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
        round_trip: bool = False,
        warnings: bool = True,
    ):
        """Usage docs: https://docs.pydantic.dev/2.5/concepts/serialization/#modelmodel_dump_json

        Generates a JSON representation of the model using Pydantic's `to_json` method.

        Args:
            indent: Indentation to use in the JSON output. If None is passed, the output will be compact.
            include: Field(s) to include in the JSON output. Can take either a string or set of strings.
            exclude: Field(s) to exclude from the JSON output. Can take either a string or set of strings.
            by_alias: Whether to serialize using field aliases.
            exclude_unset: Whether to exclude fields that have not been explicitly set.
            exclude_defaults: Whether to exclude fields that have the default value.
            exclude_none: Whether to exclude fields that have a value of `None`.
            round_trip: Whether to use serialization/deserialization between JSON and class instance.
            warnings: Whether to show any warnings that occurred during serialization.

        Returns:
            A JSON string representation of the model.
        """
        return super().model_dump_json(
            indent=indent,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            round_trip=round_trip,
            warnings=warnings
        )
        
    @classmethod
    def try_model_validate(cls, val, verbose: bool = True, none_on_fail: bool = False):
        """
        A method to try to validate a model with the given value.

        Args:
            val: The value to validate the model with.
            verbose: A boolean indicating whether to print error messages.
            none_on_fail: A boolean indicating whether to return None on failure.
        
        Returns:
            The validated model if successful, None if validation fails and none_on_fail is True, else the original value.
        """
        try:
            m: cls = try_load_model(cls, val)
            return m
        except Exception as e:
            if verbose:
                print(f"Failed to load model: {e}")
            return None if none_on_fail else val