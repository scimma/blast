import zipfile
import io
import asyncio
from typing import List, Dict, Any, Optional, Union
from django.core.files.uploadedfile import SimpleUploadedFile
from astrodash.domain.services.spectrum_service import SpectrumService
from astrodash.domain.services.classification_service import ClassificationService
from astrodash.domain.services.spectrum_processing_service import SpectrumProcessingService
from astrodash.domain.models.spectrum import Spectrum
from astrodash.config.logging import get_logger
from astrodash.core.exceptions import BatchProcessingException, ValidationException

logger = get_logger(__name__)

class BatchProcessingService:
    """
    Service for handling batch processing of spectrum files.
    Supports both zip files and individual file lists.
    """

    def __init__(
        self,
        spectrum_service: SpectrumService,
        classification_service: ClassificationService,
        processing_service: SpectrumProcessingService
    ):
        self.spectrum_service = spectrum_service
        self.classification_service = classification_service
        self.processing_service = processing_service
        self.supported_extensions = (".fits", ".dat", ".txt", ".lnw", ".csv")

    async def process_batch(
        self,
        files: Union[Any, List[Any]],
        params: Dict[str, Any],
        model_type: str,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of spectrum files.

        Args:
            files: Either a zip file (UploadFile) or a list of individual files
            params: Processing parameters
            model_type: Type of model to use ('dash', 'transformer', 'user_uploaded')
            model_id: User model ID if using user_uploaded model

        Returns:
            Dictionary with results for each file

        Raises:
            BatchProcessingException: If batch processing fails
            ValidationException: If input validation fails
        """
        try:
            # Validate input
            if files is None:
                raise ValidationException("No files provided for batch processing")

            # Check if it's a zip file (UploadedFile with .zip extension or single file)
            if hasattr(files, 'name') and hasattr(files, 'read'):
                logger.info(f"Processing file: {getattr(files, 'name', 'unknown')}")
                return await self._process_zip_file(files, params, model_type, model_id)
            elif isinstance(files, list):
                return await self._process_file_list(files, params, model_type, model_id)
            else:
                raise ValidationException(f"Invalid files type: {type(files)}. Expected UploadedFile or List[UploadedFile]")

        except (ValidationException, BatchProcessingException):
            raise
        except Exception as e:
            logger.error(f"Error in batch processing: {e}", exc_info=True)
            raise BatchProcessingException(f"Batch processing failed: {str(e)}")

    async def _process_zip_file(
        self,
        zip_file: Any,
        params: Dict[str, Any],
        model_type: str,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process files from a zip archive."""
        logger.info(f"Processing zip file: {getattr(zip_file, 'name', 'unknown')}")

        results: Dict[str, Any] = {}
        contents = zip_file.read()

        # Initialize classifier once per batch
        classifier = self.classification_service.model_factory.get_classifier(
            model_type, model_id if model_type == "user_uploaded" else None
        )

        entries: List[tuple] = []
        with zipfile.ZipFile(io.BytesIO(contents)) as zf:
            for fname in zf.namelist():
                info = zf.getinfo(fname)
                if info.is_dir():
                    continue  # Skip directories

                # Check file type support
                if not fname.lower().endswith(self.supported_extensions):
                    results[fname] = {"error": "Unsupported file type"}
                    continue

                try:
                    with zf.open(fname) as file_obj:
                        # Prepare file-like object for spectrum service
                        file_like = self._prepare_file_object(fname, file_obj)
                        entries.append((fname, file_like))
                except Exception as e:
                    logger.error(f"Error reading file {fname}: {e}")
                    results[fname] = {"error": str(e)}

        # Concurrency for processing prepared entries
        if entries:
            max_concurrency = min(8, len(entries))
            semaphore = asyncio.Semaphore(max_concurrency)

            async def worker_zip(name: str, file_like_obj: Any) -> None:
                async with semaphore:
                    try:
                        result = await self._process_single_file(
                            file_like_obj, name, params, model_type, model_id, classifier
                        )
                        results[name] = result
                    except Exception as e:
                        logger.error(f"Error processing file {name}: {e}")
                        results[name] = {"error": str(e)}

            await asyncio.gather(*(worker_zip(n, f) for n, f in entries))

        logger.info(f"Zip processing completed. Processed {len(results)} files.")
        return results

    async def _process_file_list(
        self,
        files: List[Any],
        params: Dict[str, Any],
        model_type: str,
        model_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a list of individual files."""
        # In case empty for some reason
        if not files:
            logger.warning("No files provided for processing")
            return {}

        logger.info(f"Processing {len(files)} individual files")

        results: Dict[str, Any] = {}

        # Initialize classifier once per batch
        classifier = self.classification_service.model_factory.get_classifier(
            model_type, model_id if model_type == "user_uploaded" else None
        )

        max_concurrency = min(8, max(1, len(files)))
        semaphore = asyncio.Semaphore(max_concurrency)

        async def worker(single_file: Any) -> None:
            filename_local = getattr(single_file, 'name', getattr(single_file, 'filename', 'unknown'))
            if not filename_local.lower().endswith(self.supported_extensions):
                results[filename_local] = {"error": "Unsupported file type"}
                return
            async with semaphore:
                try:
                    result_local = await self._process_single_file(
                        single_file, filename_local, params, model_type, model_id, classifier
                    )
                    results[filename_local] = result_local
                except Exception as e:
                    logger.error(f"Error processing file {filename_local}: {e}")
                    results[filename_local] = {"error": str(e)}

        await asyncio.gather(*(worker(f) for f in files))

        logger.info(f"File list processing completed. Processed {len(results)} files.")
        return results

    def _prepare_file_object(self, fname: str, file_obj) -> Any:
        """Prepare a file-like object for the spectrum service."""
        ext = fname.lower().split('.')[-1]

        if ext == 'fits':
            # For FITS files, we need to read the content once
            content = file_obj.read()
            return SimpleUploadedFile(fname, content)
        else:
            # For text files, read content once and create StringIO
            content = file_obj.read()
            try:
                text = content.decode('utf-8')
                return SimpleUploadedFile(fname, text.encode('utf-8'))
            except UnicodeDecodeError:
                return SimpleUploadedFile(fname, content)

    async def _process_single_file(
        self,
        file: Any,  # Using Any to handle starlette.datastructures.UploadFile
        filename: str,
        params: Dict[str, Any],
        model_type: str,
        model_id: Optional[str] = None,
        classifier=None
    ) -> Dict[str, Any]:
        """Process a single spectrum file."""
        try:
            # Get spectrum from file
            spectrum = await self.spectrum_service.get_spectrum_from_file(file)

            # Apply processing parameters
            processed_spectrum = await self.processing_service.process_spectrum_with_params(
                spectrum, params
            )

            # Classify with appropriate model
            if model_type == "user_uploaded":
                result = await self.classification_service.classify_spectrum(
                    processed_spectrum,
                    model_type="user_uploaded",
                    user_model_id=model_id,
                    classifier=classifier
                )
            else:
                result = await self.classification_service.classify_spectrum(
                    processed_spectrum,
                    model_type=model_type,
                    classifier=classifier
                )

            # Format result
            return {
                "spectrum": {
                    "x": processed_spectrum.x,
                    "y": processed_spectrum.y,
                    "redshift": getattr(processed_spectrum, 'redshift', None)
                },
                "classification": result.results,
                "model_type": model_type,
                "model_id": model_id if model_type == "user_uploaded" else None
            }

        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            raise
