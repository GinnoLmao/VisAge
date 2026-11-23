"""Additional upload formats preserve admission limits and legacy pixels."""
import base64
from dataclasses import replace
from io import BytesIO
import numpy as np
from PIL import Image
import pytest
from visage.faces import FaceConfig, FaceError, decode_image
from visage.image_formats import decode_upload_image, converted_preview


def encode(image_format, image=None, **kwargs):
    with BytesIO() as stream:
        if image is None:
            with Image.new('RGB', (100, 120), (40, 100, 170)) as source:
                source.save(stream, format=image_format, **kwargs)
        else:
            image.save(stream, format=image_format, **kwargs)
        return stream.getvalue()


@pytest.mark.parametrize('image_format', ['JPEG', 'PNG', 'WEBP'])
def test_legacy_uploads_keep_exact_frozen_decoder_pixels(image_format):
    payload = encode(image_format)
    with decode_upload_image(payload) as actual, decode_image(payload) as expected:
        np.testing.assert_array_equal(actual, expected)
        assert converted_preview(actual) is None


@pytest.mark.parametrize('image_format', ['HEIF', 'AVIF', 'BMP', 'TIFF'])
def test_additional_still_formats_decode_and_receive_a_bounded_jpeg_preview(image_format):
    with decode_upload_image(encode(image_format)) as image:
        assert image.mode == 'RGB' and image.size == (100, 120)
        preview = converted_preview(image)
        assert preview['content_type'] == 'image/jpeg'
        with Image.open(BytesIO(base64.b64decode(preview['data_base64']))) as browser:
            assert browser.format == 'JPEG' and browser.size == image.size


def test_additional_formats_enforce_limits_and_refuse_multiframe_files():
    data = encode('BMP')
    for config, code in [(replace(FaceConfig(), max_bytes=100), 'IMAGE_TOO_LARGE'),
                         (replace(FaceConfig(), max_pixels=100), 'IMAGE_DIMENSIONS_TOO_LARGE')]:
        with pytest.raises(FaceError) as error:
            decode_upload_image(data, config)
        assert error.value.code == code
    with Image.new('RGB', (100, 100)) as image, Image.new('RGB', (100, 100), 'red') as other:
        multiple = encode('TIFF', image, save_all=True, append_images=[other])
    with pytest.raises(FaceError) as error:
        decode_upload_image(multiple)
    assert error.value.code == 'ANIMATED_IMAGE'
    with pytest.raises(FaceError) as error:
        decode_upload_image(encode('GIF'))
    assert error.value.code == 'UNSUPPORTED_IMAGE'


def test_extra_format_exif_orientation_and_preview_match_selected_coordinates():
    with Image.new('RGB', (100, 140), 'blue') as image:
        exif = image.getexif(); exif[274] = 6
        payload = encode('TIFF', image, exif=exif)
    with decode_upload_image(payload) as decoded:
        assert decoded.size == (140, 100)
        preview = converted_preview(decoded)
        assert (preview['width'], preview['height']) == decoded.size
