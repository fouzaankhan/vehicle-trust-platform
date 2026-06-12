import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VehicleImageAnalyzer:
    """
    OpenCV-only image analysis pipeline.
    No deep learning dependency — fully lightweight and portable.
    """

    def analyze(self, image_path: str) -> dict:
        img = self._load_image(image_path)
        if img is None:
            return {"error": f"Could not load image: {image_path}"}

        quality = self._assess_quality(img)
        damage = self._estimate_damage_signals(img)
        composition = self._assess_composition(img)
        damage_flags = self._damage_flags(damage)

        condition_score = self._combine_scores(quality, damage)

        return {
            "quality_metrics": quality,
            "damage_signals": damage,
            "damage_flags": damage_flags,
            "composition": composition,
            "image_condition_score": condition_score,
            "verdict": self._verdict(condition_score, quality)
        }

    def _load_image(self, image_path: str):
        if not os.path.exists(image_path):
            logger.error(f"File not found: {image_path}")
            return None
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"cv2 could not read: {image_path}")
            return None
        # Resize for consistent processing
        img = cv2.resize(img, (800, 600))
        return img

    # ------------------------------------------------------------------ #
    # Quality: blur, brightness, contrast
    # ------------------------------------------------------------------ #
    def _assess_quality(self, img) -> dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur detection — Laplacian variance. Higher = sharper.
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Brightness — mean pixel intensity (0-255)
        brightness = float(np.mean(gray))

        # Contrast — std deviation of pixel intensities
        contrast = float(np.std(gray))

        return {
            "blur_score": round(float(blur_score), 2),
            "brightness": round(brightness, 2),
            "contrast": round(contrast, 2),
            "is_blurry": bool(blur_score < 100),
            "is_too_dark": bool(brightness < 60),
            "is_too_bright": bool(brightness > 200)
        }

    # ------------------------------------------------------------------ #
    # Damage signals: rust detection via HSV color filtering
    # ------------------------------------------------------------------ #
    def _estimate_damage_signals(self, img) -> dict:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Step 1: Color-based candidate mask (orange-brown tones)
        rust_lower = np.array([5, 50, 50])
        rust_upper = np.array([20, 255, 200])
        rust_color_mask = cv2.inRange(hsv, rust_lower, rust_upper)
        rust_color_ratio = float(rust_color_mask.sum()) / (img.shape[0] * img.shape[1] * 255)

        # Step 2: Texture check — rust is rough/noisy, smooth tan paint is not.
        # Compute local variance (texture) only within the candidate region.
        rust_texture_score = 0.0
        if rust_color_ratio > 0.01:
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            masked_pixels = lap[rust_color_mask > 0]
            if len(masked_pixels) > 0:
                rust_texture_score = float(np.var(masked_pixels))

        # Rust = correct color AND rough texture
        rust_detected = bool(rust_color_ratio > 0.04 and rust_texture_score > 800)

        # Edge density — proxy for dents/scratches
        edges = cv2.Canny(gray, 100, 200)
        edge_density = float(edges.mean())

        # Dark patch detection — potential dents/shadows
        dark_mask = cv2.inRange(gray, 0, 40)
        dark_ratio = float(dark_mask.sum()) / (img.shape[0] * img.shape[1] * 255)

        return {
            "rust_color_ratio_pct": round(rust_color_ratio * 100, 2),
            "rust_texture_score": round(rust_texture_score, 2),
            "edge_density": round(edge_density, 2),
            "dark_patch_ratio_pct": round(dark_ratio * 100, 2),
            "rust_detected": rust_detected,
            "high_edge_complexity": bool(edge_density > 30)
        }

    # ------------------------------------------------------------------ #
    # Composition: is this a proper vehicle photo or random/stock image?
    # ------------------------------------------------------------------ #
    def _assess_composition(self, img) -> dict:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Color variance — stock/studio photos often have very uniform backgrounds
        color_std = float(np.std(img))

        # Aspect ratio check (after resize this is always 800x600, so check original)
        h, w = img.shape[:2]
        aspect_ratio = round(w / h, 2)

        return {
            "color_variance": round(color_std, 2),
            "aspect_ratio": aspect_ratio,
            "likely_studio_photo": bool(color_std < 30)
        }

    # ------------------------------------------------------------------ #
    # Combine into single 0-100 condition score
    # ------------------------------------------------------------------ #
    def _combine_scores(self, quality, damage) -> float:
        """
        Score is driven primarily by IMAGE QUALITY — which is reliable.
        Damage/rust signals are informational only (see _verdict) because
        color/texture heuristics without object localization are unreliable
        on uncropped images (background, ground, bumpers create false positives).
        """
        score = 50.0  # neutral baseline

        if quality["is_blurry"]:
            score -= 30
        else:
            score += 20

        if quality["is_too_dark"] or quality["is_too_bright"]:
            score -= 15
        else:
            score += 10

        # Small contrast bonus — sharp, well-lit photos score higher
        if quality["contrast"] > 50:
            score += 10

        return round(max(0, min(100, score)), 1)

    def _verdict(self, condition_score, quality) -> str:
        if quality["is_blurry"]:
            return "IMAGE QUALITY TOO LOW — request clearer photos"
        elif condition_score >= 70:
            return "GOOD IMAGE QUALITY — suitable for listing review"
        elif condition_score >= 45:
            return "ACCEPTABLE IMAGE QUALITY — minor issues"
        else:
            return "POOR IMAGE QUALITY — request better photos"
        
    def _damage_flags(self, damage) -> dict:
        flags = []

        if damage["rust_detected"]:
            flags.append(
                "Possible rust/discoloration detected — verify visually"
            )

        if damage["high_edge_complexity"]:
            flags.append(
                "High visual complexity — possible damage or busy background"
            )

        if damage["dark_patch_ratio_pct"] > 25:
            flags.append(
                "Large dark regions — possible shadow or dent"
            )

        return {
            "flags": flags if flags else ["No flags raised"],
            "reliability": "EXPERIMENTAL — requires manual verification"
        }


if __name__ == "__main__":
    analyzer = VehicleImageAnalyzer()

    test_dir = "data/sample_images"
    if os.path.exists(test_dir):
        for filename in os.listdir(test_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(test_dir, filename)
                result = analyzer.analyze(path)
                print(f"\n{'='*50}")
                print(f"FILE: {filename}")
                print(f"{'='*50}")
                import json
                print(json.dumps(result, indent=2))
    else:
        print(f"Create {test_dir} and add test images first.")