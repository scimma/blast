from __future__ import annotations

import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_http_methods
from django.utils.decorators import method_decorator
from django.core.files.uploadedfile import UploadedFile
from asgiref.sync import async_to_sync

from astrodash.config.logging import get_logger
from astrodash.core.exceptions import (
    AppException,
    ValidationException,
    TemplateNotFoundException,
    ElementNotFoundException,
    SpectrumProcessingException,
)
from astrodash.services import (
    get_template_analysis_service,
    get_line_list_service,
    get_spectrum_processing_service,
    get_spectrum_service,
    get_classification_service,
    get_model_service,
    get_batch_processing_service,
    get_redshift_service,
)
from astrodash.shared.utils.helpers import sanitize_for_json, construct_osc_reference

logger = get_logger(__name__)


def _json_error(message: str, status: int = 400):
    return JsonResponse({"detail": message}, status=status)


def _parse_params(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise ValidationException("Invalid JSON supplied for params")


@require_GET
def analysis_options(request):
    service = get_template_analysis_service()
    result = async_to_sync(service.get_analysis_options)()
    return JsonResponse(result)


@require_GET
def template_statistics(request):
    service = get_template_analysis_service()
    result = async_to_sync(service.get_template_statistics)()
    return JsonResponse(result)


@require_GET
def template_spectrum(request):
    sn_type = request.GET.get("sn_type", "Ia")
    age_bin = request.GET.get("age_bin", "2 to 6")
    service = get_template_analysis_service()
    try:
        wave, flux = service.template_handler.get_template_spectrum(sn_type, age_bin)
        return JsonResponse({"x": wave.tolist(), "y": flux.tolist()})
    except TemplateNotFoundException as exc:
        return _json_error(exc.message, status=exc.status_code)


@require_GET
def template_line_list(request):
    service = get_line_list_service()
    try:
        return JsonResponse(service.get_line_list())
    except AppException as exc:
        return _json_error(exc.message, status=exc.status_code)


@require_GET
def line_list_elements(request):
    service = get_line_list_service()
    return JsonResponse({"elements": service.get_available_elements()})


@require_GET
def line_list_element(request, element: str):
    service = get_line_list_service()
    try:
        return JsonResponse({
            "element": element,
            "wavelengths": service.get_element_wavelengths(element)
        })
    except ElementNotFoundException as exc:
        return _json_error(exc.message, status=exc.status_code)


@require_GET
def line_list_filter(request):
    try:
        min_wave = float(request.GET.get("min_wavelength"))
        max_wave = float(request.GET.get("max_wavelength"))
    except (TypeError, ValueError):
        return _json_error("min_wavelength and max_wavelength query params are required", status=400)

    if min_wave > max_wave:
        return _json_error("Minimum wavelength must be <= maximum wavelength")

    service = get_line_list_service()
    filtered = service.filter_wavelengths_by_range(min_wave, max_wave)
    return JsonResponse({
        "min_wavelength": min_wave,
        "max_wavelength": max_wave,
        "elements": filtered,
        "count": len(filtered),
    })


@csrf_exempt
@require_http_methods(["POST"])
def process_spectrum(request):
    try:
        params = _parse_params(request.POST.get("params"))
    except ValidationException as exc:
        return _json_error(exc.message, status=exc.status_code)

    model_id = request.POST.get("model_id")
    uploaded_file = request.FILES.get("file")

    model_type = "user_uploaded" if model_id else params.get("modelType", "dash")
    if model_type not in ("dash", "transformer", "user_uploaded"):
        model_type = "dash"

    osc_ref = params.get("oscRef")
    if osc_ref:
        params["oscRef"] = construct_osc_reference(osc_ref)
        osc_ref = params["oscRef"]

    spectrum_service = get_spectrum_service()
    processing_service = get_spectrum_processing_service()
    classification_service = get_classification_service()

    try:
        spectrum = async_to_sync(spectrum_service.get_spectrum_data)(
            file=uploaded_file,
            osc_ref=osc_ref,
        )
        try:
            async_to_sync(spectrum_service.save_spectrum)(spectrum)
        except Exception as exc:  # noqa: BLE001 - best effort persistence
            logger.warning("Unable to persist spectrum: %s", exc)

        processed = async_to_sync(processing_service.process_spectrum_with_params)(
            spectrum=spectrum,
            params=params,
        )

        classification = async_to_sync(classification_service.classify_spectrum)(
            spectrum=processed,
            model_type=model_type,
            user_model_id=model_id,
            params=params,
        )

        payload = {
            "spectrum": {
                "x": processed.x,
                "y": processed.y,
                "redshift": processed.redshift,
            },
            "classification": sanitize_for_json(classification.results),
            "model_type": classification.model_type,
            "model_id": model_id,
        }
        return JsonResponse(payload)

    except AppException as exc:
        return _json_error(exc.message, status=exc.status_code)
    except Exception as exc:  # noqa: BLE001 - generic failure path
        logger.error("Spectrum processing failed", exc_info=True)
        return _json_error(f"Processing error: {exc}", status=500)


@csrf_exempt
@require_http_methods(["POST"])
def estimate_redshift(request):
    file_obj = request.FILES.get("file")
    sn_type = request.POST.get("sn_type")
    age_bin = request.POST.get("age_bin")

    if not file_obj or not sn_type or not age_bin:
        return _json_error("file, sn_type, and age_bin are required", status=400)

    try:
        contents = file_obj.read().decode("utf-8")
    except UnicodeDecodeError:
        return _json_error("Uploaded file must be UTF-8 encoded text", status=400)

    x_vals, y_vals = [], []
    for raw in contents.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            x_vals.append(float(parts[0]))
            y_vals.append(float(parts[1]))
        except ValueError:
            continue

    if not x_vals or not y_vals:
        return _json_error("No valid spectrum data found in file", status=400)

    redshift_service = get_redshift_service()
    result = async_to_sync(redshift_service.estimate_redshift_from_spectrum)(
        x_vals, y_vals, sn_type, age_bin, model_type="dash"
    )
    return JsonResponse(sanitize_for_json(result))


@csrf_exempt
@require_http_methods(["POST"])
def upload_model(request):
    model_file = request.FILES.get("file")
    class_mapping = request.POST.get("class_mapping")
    input_shape = request.POST.get("input_shape")
    name = request.POST.get("name", "")
    description = request.POST.get("description", "")
    owner = request.POST.get("owner", "")

    if not model_file or not class_mapping or not input_shape:
        return _json_error("file, class_mapping, and input_shape are required", status=400)

    service = get_model_service()
    try:
        user_model, model_info = async_to_sync(service.upload_model)(
            model_content=model_file.read(),
            filename=model_file.name,
            class_mapping_str=class_mapping,
            input_shape_str=input_shape,
            name=name or model_file.name,
            description=description,
            owner=owner,
        )
        response = {
            "status": "success",
            "message": "Model uploaded and validated successfully.",
            "model_id": user_model.id,
            "model_filename": model_file.name,
            "class_mapping": model_info.get("class_mapping"),
            "output_shape": model_info.get("output_shape"),
            "input_shape": model_info.get("input_shapes", [None])[0] if model_info.get("input_shapes") else None,
            "model_info": sanitize_for_json(model_info),
        }
        return JsonResponse(response)
    except AppException as exc:
        return _json_error(exc.message, status=exc.status_code)
    except Exception as exc:  # noqa: BLE001
        logger.error("Model upload failed", exc_info=True)
        return _json_error(f"Internal server error: {exc}", status=500)


@require_GET
def list_models(request):
    service = get_model_service()
    models = async_to_sync(service.list_models)()
    payload = []
    for model in models:
        try:
            info = service.get_model_info(model.id)
        except Exception:  # noqa: BLE001
            info = {}
        payload.append({
            "id": model.id,
            "name": model.name,
            "owner": model.owner,
            "description": model.description,
            "model_filename": model.name,
            "class_mapping": info.get("class_mapping"),
            "input_shape": info.get("input_shape"),
            "meta": model.meta,
        })
    return JsonResponse(payload, safe=False)


@require_GET
def get_model_info_view(request, model_id: str):
    model_id = str(model_id)
    service = get_model_service()
    try:
        info = service.get_model_info(model_id)
        return JsonResponse(sanitize_for_json(info))
    except AppException as exc:
        return _json_error(exc.message, status=exc.status_code)


@csrf_exempt
@require_http_methods(["DELETE"])
def delete_model(request, model_id: str):
    model_id = str(model_id)
    service = get_model_service()
    try:
        async_to_sync(service.delete_model)(model_id)
        return JsonResponse({"status": "success", "message": f"Model {model_id} deleted successfully."})
    except AppException as exc:
        return _json_error(exc.message, status=exc.status_code)


@csrf_exempt
@require_http_methods(["PUT"])
def update_model(request, model_id: str):
    model_id = str(model_id)
    service = get_model_service()
    if request.body:
        try:
            data = json.loads(request.body.decode("utf-8"))
        except json.JSONDecodeError:
            return _json_error("Invalid JSON payload", status=400)
    else:
        data = {}
    try:
        updated = async_to_sync(service.update_model_metadata)(model_id, data)
        return JsonResponse({"status": "success", "model_id": updated.id})
    except AppException as exc:
        return _json_error(exc.message, status=exc.status_code)


@require_GET
def list_models_by_owner(request, owner: str):
    service = get_model_service()
    try:
        models = async_to_sync(service.list_models_by_owner)(owner)
        payload = [
            {
                "id": model.id,
                "name": model.name,
                "owner": model.owner,
                "description": model.description,
                "meta": model.meta,
            }
            for model in models
        ]
        return JsonResponse(payload, safe=False)
    except AppException as exc:
        return _json_error(exc.message, status=exc.status_code)


@csrf_exempt
@require_http_methods(["POST"])
def batch_process(request):
    try:
        params = _parse_params(request.POST.get("params"))
    except ValidationException as exc:
        return _json_error(exc.message, status=exc.status_code)

    zip_file = request.FILES.get("zip_file")
    files = request.FILES.getlist("files")
    model_id = request.POST.get("model_id")

    if zip_file and files:
        return _json_error("Provide either zip_file or files, not both", status=400)
    if not zip_file and not files:
        return _json_error("No files provided", status=400)

    model_type = "user_uploaded" if model_id else params.get("modelType", "dash")
    if model_type not in ("dash", "transformer", "user_uploaded"):
        model_type = "dash"

    service = get_batch_processing_service()
    try:
        payload = zip_file if zip_file else files
        results = async_to_sync(service.process_batch)(
            payload,
            params,
            model_type,
            model_id,
        )
        return JsonResponse(sanitize_for_json(results))
    except AppException as exc:
        return _json_error(exc.message, status=exc.status_code)
    except Exception as exc:
        logger.error("Batch processing failed", exc_info=True)
        return _json_error(f"Batch process error: {exc}", status=500)
