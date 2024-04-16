from pathlib import Path
from tempfile import NamedTemporaryFile
from datamodel_code_generator import generate, DataModelType


def json_to_model(
    json_str: str,
    output_path: Path | str = None,
    class_name: str | None = None,
    data_model_type: DataModelType = DataModelType.PydanticV2BaseModel,
    validation=True,
    use_annotated=True,
    field_constraints=True,
    use_field_description=True,
    force_optional_for_required_fields=True,
    **kwargs
):
    if output_path is None:
        f = NamedTemporaryFile("w+", suffix=".py")
        output_path = Path(f.name)
    else:
        f = None
        output_path: Path = Path(output_path)
        output_path = output_path.with_suffix(".py")
    
    generate(
        input_=json_str,
        output=output_path,
        class_name=class_name,
        validation=validation,
        use_annotated=use_annotated,
        field_constraints=field_constraints,
        use_field_description=use_field_description,
        output_model_type=data_model_type,
        force_optional_for_required_fields=force_optional_for_required_fields,
        **kwargs
    )
    output = None
    if f is not None:
        try:
            f.seek(0)
            output = f.read()
            f.close()
        except:
            pass
    return output