import numpy as np

def apply_shift(X, shift_type, severity):
    """
    Apply a synthetic shift to the data X.
    
    Args:
        X (np.array): Input feature matrix of shape (n_samples, n_features).
        shift_type (str): Type of shift to apply. Options: "gaussian", "mask", "scaling".
        severity (float): Severity level of the corruption (e.g., 0.1 for 10%).
    
    Returns:
        np.array: The shifted feature matrix.
    """
    if shift_type == "gaussian":
        # Add Gaussian noise proportional to the per-feature standard deviation.
        noise = np.random.normal(loc=0, scale=severity * np.std(X, axis=0), size=X.shape)
        return X + noise
    elif shift_type == "mask":
        # Randomly zero out a fraction of the features.
        mask = np.random.rand(*X.shape) < severity
        X_shifted = X.copy()
        X_shifted[mask] = 0
        return X_shifted
    elif shift_type == "scaling":
        # Scale all features by a factor (e.g., 1 + severity).
        scaling_factor = 1 + severity
        return X * scaling_factor
    else:
        raise ValueError(f"Unknown shift type: {shift_type}")
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2

def _to_pil(image):
    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    # If the image is grayscale (has a trailing channel of size 1), squeeze and use mode "L".
    if image_uint8.ndim == 3 and image_uint8.shape[-1] == 1:
        image_uint8 = np.squeeze(image_uint8, axis=-1)
        return Image.fromarray(image_uint8, mode="L")
    return Image.fromarray(image_uint8)

def _from_pil(pil_img):
    """Convert a PIL Image to a NumPy array with values in [0,1]."""
    arr = np.array(pil_img).astype(np.float32) / 255.0
    # If grayscale, ensure we have a channel dimension.
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr

### --- Regression Shifts --- ###
def regression_gaussian(X, severity):
    noise = np.random.normal(loc=0, scale=severity * np.std(X, axis=0), size=X.shape)
    return X + noise

def regression_mask(X, severity):
    mask = np.random.rand(*X.shape) < severity
    X_shifted = X.copy()
    X_shifted[mask] = 0
    return X_shifted

def regression_scaling(X, severity):
    scaling_factor = 1 + severity
    return X * scaling_factor

### --- Classification/Image Corruptions --- ###
def glass_blur_vectorized(image, severity, iterations=5, window=3):
    # Get image dimensions
    H, W = image.shape[:2]
    # Create coordinate grids
    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    
    # For each iteration, generate random displacement fields (vectorized)
    output = image.copy()
    for _ in range(iterations):
        # Generate random displacement fields in the range [-window, window]
        disp_x = np.random.randint(-window, window + 1, size=(H, W)).astype(np.float32)
        disp_y = np.random.randint(-window, window + 1, size=(H, W)).astype(np.float32)
        
        # Optionally smooth the displacement fields to create smoother distortions.
        # The sigma value here can be linked to the severity.
        sigma = severity  # adjust as needed
        disp_x = cv2.GaussianBlur(disp_x, (window * 2 + 1, window * 2 + 1), sigmaX=sigma)
        disp_y = cv2.GaussianBlur(disp_y, (window * 2 + 1, window * 2 + 1), sigmaX=sigma)
        
        # Compute the remapping coordinates.
        map_x = (grid_x + disp_x).astype(np.float32)
        map_y = (grid_y + disp_y).astype(np.float32)
        
        # Apply the remapping.
        output = cv2.remap(output, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    return np.clip(output, 0, 1)


def add_impulse_noise(image, severity):
    prob = severity * 0.05  # Adjust probability as needed
    rnd = np.random.rand(*image.shape)
    noisy = image.copy()
    noisy[rnd < (prob / 2)] = 0
    noisy[rnd > 1 - (prob / 2)] = 1
    return noisy

def apply_pixelate(image, severity):
    h, w, c = image.shape
    factor = 1 + severity  # Higher severity gives lower resolution
    new_h, new_w = h // factor, w // factor
    small = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.clip(pixelated, 0, 1)

def adjust_saturate(image, severity):
    # Adjust saturation using PIL's ImageEnhance.Color.
    pil_img = _to_pil(image)
    # For grayscale images, saturation does not apply—return unchanged.
    if pil_img.mode == "L":
        return _from_pil(pil_img)
    enhancer = ImageEnhance.Color(pil_img)
    # A factor < 1 desaturates; >1 increases saturation.
    # Map severity so that severity=0 gives factor=1 and increasing severity desaturates.
    factor = 1 - severity * 0.1  # adjust multiplier as needed
    saturated = enhancer.enhance(factor)
    return _from_pil(saturated)

def adjust_brightness(image, severity):
    pil_img = _to_pil(image)
    enhancer = ImageEnhance.Brightness(pil_img)
    factor = 1 + severity * 0.2  # Adjust as needed
    brightened = enhancer.enhance(factor)
    return _from_pil(brightened)

def adjust_contrast(image, severity):
    pil_img = _to_pil(image)
    enhancer = ImageEnhance.Contrast(pil_img)
    factor = 1 + severity * 0.2
    contrasted = enhancer.enhance(factor)
    return _from_pil(contrasted)

def apply_defocus_blur(image, severity):
    pil_img = _to_pil(image)
    radius = severity * 1.0  # Map severity to blur radius
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return _from_pil(blurred)

def apply_elastic_transform(image, severity):
    random_state = np.random.RandomState(None)
    # If image is grayscale, work with 2D.
    if image.ndim == 3 and image.shape[-1] == 1:
        image_2d = image.squeeze(-1)
    elif image.ndim == 2:
        image_2d = image
    else:
        image_2d = image
    shape = image_2d.shape
    alpha = severity * 20
    sigma = severity * 4
    dx = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha
    dy = cv2.GaussianBlur((random_state.rand(*shape) * 2 - 1).astype(np.float32), (17, 17), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    transformed = cv2.remap(image_2d, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if image.ndim == 3 and image.shape[-1] == 1:
        transformed = np.expand_dims(transformed, axis=-1)
    return np.clip(transformed, 0, 1)

def add_shot_noise(image, severity):
    image_clipped = np.clip(image, 0, 1)
    vals = len(np.unique(image_clipped))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image_clipped * vals) / float(vals)
    return np.clip(noisy, 0, 1)

def add_spatter(image, severity):
    if image.ndim == 3 and image.shape[-1] == 1:
        image_2d = image.squeeze(-1)
        h, w = image_2d.shape
        spatter = image_2d.copy()
        num_blobs = int(severity * 10)
        for _ in range(num_blobs):
            center = (np.random.randint(0, w), np.random.randint(0, h))
            radius = np.random.randint(2, 5 + severity)
            cv2.circle(spatter, center, radius, 1, -1)
        spatter = np.expand_dims(spatter, axis=-1)
    elif image.ndim == 2:
        h, w = image.shape
        spatter = image.copy()
        num_blobs = int(severity * 10)
        for _ in range(num_blobs):
            center = (np.random.randint(0, w), np.random.randint(0, h))
            radius = np.random.randint(2, 5 + severity)
            cv2.circle(spatter, center, radius, 1, -1)
        spatter = spatter[..., np.newaxis]
    else:
        h, w, c = image.shape
        spatter = image.copy()
        num_blobs = int(severity * 10)
        for _ in range(num_blobs):
            center = (np.random.randint(0, w), np.random.randint(0, h))
            radius = np.random.randint(2, 5 + severity)
            cv2.circle(spatter, center, radius, (1, 1, 1), -1)
    return np.clip(spatter, 0, 1)

def add_speckle_noise(image, severity):
    # Speckle noise: multiplicative noise
    noise = np.random.randn(*image.shape) * severity * 0.1
    return np.clip(image + image * noise, 0, 1)

def apply_zoom_blur(image, severity):
    # Handle grayscale separately
    if image.ndim == 3 and image.shape[-1] == 1:
        image_2d = image.squeeze(-1)
        h, w = image_2d.shape
        zoom_factor = 1 + severity * 0.1
        zoomed = cv2.resize(image_2d, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        zh, zw = zoomed.shape
        start_h = (zh - h) // 2
        start_w = (zw - w) // 2
        cropped = zoomed[start_h:start_h+h, start_w:start_w+w]
        blended = (image_2d + cropped) / 2
        blended = np.expand_dims(blended, axis=-1)
    else:
        h, w, c = image.shape
        zoom_factor = 1 + severity * 0.1
        zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
        zh, zw, _ = zoomed.shape
        start_h = (zh - h) // 2
        start_w = (zw - w) // 2
        cropped = zoomed[start_h:start_h+h, start_w:start_w+w, :]
        blended = (image + cropped) / 2
    return np.clip(blended, 0, 1)

def add_fog(image, severity):
    fog = np.full(image.shape, 1.0)
    alpha = severity * 0.1
    foggy = image * (1 - alpha) + fog * alpha
    return np.clip(foggy, 0, 1)

def add_frost(image, severity):
    if image.ndim == 3 and image.shape[-1] == 1:
        image_2d = image.squeeze(-1)
        h, w = image_2d.shape
        frost = np.random.rand(h, w)
        mask = frost < (severity * 0.1)
        frosted = image_2d.copy()
        frosted[mask] = 1.0
        return np.expand_dims(frosted, axis=-1)
    elif image.ndim == 2:
        frost = np.random.rand(*image.shape)
        mask = frost < (severity * 0.1)
        frosted = image.copy()
        frosted[mask] = 1.0
        return frosted[..., np.newaxis]
    else:
        h, w, c = image.shape
        frost = np.random.rand(h, w)
        mask = frost < (severity * 0.1)
        frosted = image.copy()
        frosted[mask] = 1.0
        return frosted

def apply_gaussian_blur(image, severity):
    # New function for Gaussian Blur distinct from defocus blur.
    pil_img = _to_pil(image)
    radius = severity * 0.5  # Different mapping from defocus blur
    blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    return _from_pil(blurred)

def add_gaussian_noise(image, severity):
    std = severity * 0.08  # Adjust multiplier as needed
    noise = np.random.normal(loc=0.0, scale=std, size=image.shape)
    return np.clip(image + noise, 0, 1)

### --- Unified apply_shift --- ###
def apply_shift(X, shift_type, severity):
    """
    Apply a synthetic shift to a batch of inputs.
    
    For regression tasks (tabular data), use simple numeric shifts:
      - "gaussian", "mask", "scaling"
      
    For classification/image tasks, if the shift_type is not a regression type,
    assume it is one of the 16 image corruption types:
    
    1. Glass Blur
    2. Impulse Noise
    3. Pixelate
    4. Saturate
    5. Brightness
    6. Contrast
    7. Defocus Blur
    8. Elastic Transform
    9. Shot noise
    10. Spatter
    11. Speckle noise
    12. Zoom blur
    13. Fog
    14. Frost
    15. Gaussian Blur
    16. Gaussian noise
    """
    regression_shifts = ["gaussian", "mask", "scaling"]
    if shift_type in regression_shifts:
        if shift_type == "gaussian":
            return regression_gaussian(X, severity)
        elif shift_type == "mask":
            return regression_mask(X, severity)
        elif shift_type == "scaling":
            return regression_scaling(X, severity)
    else:
        original_shape = X.shape
        # If input is flattened, reshape appropriately.
        if X.ndim == 2:
            n = X.shape[0]
            num_pixels = X.shape[1]
            if num_pixels == 1024:
                # Grayscale: 32x32 with 1 channel.
                X = X.reshape(n, 32, 32, 1)
            elif num_pixels == 3072:
                # Color: 32x32 with 3 channels.
                X = X.reshape(n, 32, 32, 3)
            else:
                raise ValueError(f"Unexpected flattened image size: {num_pixels}")
        corrupted_images = []
        for img in X:
            if shift_type == "Glass Blur":
                img_corrupted = glass_blur_vectorized(img, severity)
            elif shift_type == "Impulse Noise":
                img_corrupted = add_impulse_noise(img, severity)
            elif shift_type == "Pixelate":
                img_corrupted = apply_pixelate(img, severity)
            elif shift_type == "Saturate":
                img_corrupted = adjust_saturate(img, severity)
            elif shift_type == "Brightness":
                img_corrupted = adjust_brightness(img, severity)
            elif shift_type == "Contrast":
                img_corrupted = adjust_contrast(img, severity)
            elif shift_type == "Defocus Blur":
                img_corrupted = apply_defocus_blur(img, severity)
            elif shift_type == "Elastic Transform":
                img_corrupted = apply_elastic_transform(img, severity)
            elif shift_type.lower() == "shot noise":  # allow case-insensitive match
                img_corrupted = add_shot_noise(img, severity)
            elif shift_type == "Spatter":
                img_corrupted = add_spatter(img, severity)
            elif shift_type.lower() == "speckle noise":  # allow case-insensitive match
                img_corrupted = add_speckle_noise(img, severity)
            elif shift_type.lower() == "zoom blur" or shift_type.lower() == "zoom blur":
                img_corrupted = apply_zoom_blur(img, severity)
            elif shift_type == "Fog":
                img_corrupted = add_fog(img, severity)
            elif shift_type == "Frost":
                img_corrupted = add_frost(img, severity)
            elif shift_type == "Gaussian Blur":
                img_corrupted = apply_gaussian_blur(img, severity)
            elif shift_type == "Gaussian noise":
                img_corrupted = add_gaussian_noise(img, severity)
            else:
                raise ValueError(f"Unknown or unsupported shift type: {shift_type}")
            corrupted_images.append(img_corrupted)
        corrupted_images = np.array(corrupted_images)
        # If the original input was flattened, flatten the corrupted images back.
        if len(original_shape) == 2:
            corrupted_images = corrupted_images.reshape(original_shape[0], -1)
        return corrupted_images
