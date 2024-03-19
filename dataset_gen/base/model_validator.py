from models.base import CustomBaseModel as BaseModel


class BaseModelValidator:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _to_model(self, val: str | dict, model_class=BaseModel):
        try:
            if isinstance(val, str):
                func = model_class.model_validate_json
            else:
                func = model_class.model_validate
            return func(val)
        except Exception as e:
            if self.verbose:
                print(f"Failed to load model: {e}")

    def _validate(
        self, val: str | dict, model_class=BaseModel, return_none_on_error: bool = False
    ) -> str | dict | None:
        model = self._to_model(val, model_class)
        return (
            model.model_dump(mode="json", by_alias=True)
            if isinstance(model, model_class)
            else None if return_none_on_error else val
        )
