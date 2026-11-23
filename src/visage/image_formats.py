"""Additional still-image admission without altering the frozen JPEG pipeline."""
import base64
from io import BytesIO
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError
from pillow_heif import register_heif_opener
from .faces import FaceConfig, FaceError, decode_image

register_heif_opener(thumbnails=False)
LEGACY_FORMATS = {'JPEG', 'PNG', 'WEBP'}
ADDITIONAL_FORMATS = {'HEIF', 'AVIF', 'BMP', 'TIFF'}
IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/webp', 'image/heic', 'image/heif',
    'image/heic-sequence', 'image/heif-sequence', 'image/avif', 'image/bmp', 'image/x-ms-bmp', 'image/tiff', 'application/octet-stream'}
FORMAT_DESCRIPTION = 'JPEG, PNG, WebP, HEIC/HEIF, AVIF, BMP or a single-page TIFF'


def decode_upload_image(data, config=None):
    config = config or FaceConfig()
    if not isinstance(data, bytes) or not data or len(data) > config.max_bytes:
        return decode_image(data, config)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.format in LEGACY_FORMATS:
                    return decode_image(data, config)
                if source.format not in ADDITIONAL_FORMATS:
                    raise FaceError('UNSUPPORTED_IMAGE', f'Choose {FORMAT_DESCRIPTION}.', status_code=415)
                if source.width * source.height > config.max_pixels:
                    raise FaceError('IMAGE_DIMENSIONS_TOO_LARGE', 'Image exceeds the 20 megapixel decoded limit.', status_code=413)
                if getattr(source, 'n_frames', 1) != 1:
                    raise FaceError('ANIMATED_IMAGE', 'Choose one still image rather than a sequence or multipage file.', status_code=415)
                if min(source.size) < 32:
                    raise FaceError('IMAGE_TOO_SMALL', 'Image dimensions must be at least 32 pixels.')
                source.load()
                oriented = ImageOps.exif_transpose(source)
                try:
                    if oriented.mode in {'RGBA', 'LA'} or 'transparency' in oriented.info:
                        with oriented.convert('RGBA') as rgba, rgba.getchannel('A') as alpha:
                            result = Image.new('RGB', rgba.size, (127, 127, 127))
                            result.paste(rgba, mask=alpha)
                    else:
                        result = oriented.convert('RGB')
                    result.info['visage_converted_preview'] = True
                    return result
                finally:
                    if oriented is not source:
                        oriented.close()
    except FaceError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise FaceError('IMAGE_DIMENSIONS_TOO_LARGE', 'Image exceeds the decoded image safety limit.', status_code=413) from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, RuntimeError) as exc:
        raise FaceError('INVALID_IMAGE', 'Image is unreadable or corrupt.', status_code=400) from exc


def converted_preview(image):
    """A bounded, oriented browser preview; the classifier keeps original pixels."""
    if not image.info.get('visage_converted_preview'):
        return None
    with image.copy() as preview:
        preview.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
        with BytesIO() as stream:
            preview.save(stream, format='JPEG', quality=88)
            return {'content_type': 'image/jpeg', 'width': preview.width, 'height': preview.height,
                'data_base64': base64.b64encode(stream.getvalue()).decode('ascii')}
