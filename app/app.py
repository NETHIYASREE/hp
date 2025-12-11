import os
import io
import pickle
import numpy as np
from flask import Flask, render_template, request, url_for, send_file, jsonify
from PIL import Image, ImageDraw, ImageFont

app = Flask(__name__)

# ---- FEATURES (order must match how model was trained) ----
FEATURES = [
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "parking",
    "prefarea",
    "furnishingstatus",
]

# ---- Load model (robust path resolution) ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")  # app/model/model.pkl

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def parse_value(name, raw_value):
    """Convert incoming form value into a float matching training encoding."""
    if raw_value is None:
        raise ValueError(f"Missing input for: {name}")

    raw = str(raw_value).strip()
    if raw == "":
        raise ValueError(f"Missing input for: {name}")

    if name == "furnishingstatus":
        mapping = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
        try:
            return float(raw)
        except ValueError:
            key = raw.lower()
            if key in mapping:
                return float(mapping[key])
            raise ValueError(f"Unrecognized furnishingstatus: {raw}")

    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Invalid numeric input for {name}: {raw}")


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")
    


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = []
        for feat in FEATURES:
            raw = request.form.get(feat)
            val = parse_value(feat, raw)
            values.append(val)

        X = np.array(values).reshape(1, -1)
        pred = model.predict(X)[0]

        # Format prediction (example: convert to lakhs). Adjust if your model outputs differently.
        try:
            lakhs = float(pred) / 100000.0
            prediction_text = f"{lakhs:.2f} L"
        except Exception:
            prediction_text = f"{float(pred):.2f}"

        # Pass total area to template so split page can use it
        try:
            total_area = float(request.form.get("area"))
        except Exception:
            total_area = None

        return render_template("estimate.html", prediction=prediction_text, total_area=total_area)

    except Exception as e:
        # keep helpful debug information during development
        return render_template("index.html", prediction=f"Error: {e}")


@app.route("/split", methods=["GET"])
def split():
    """
    Renders the SQFT split-up page.
    Accepts optional ?area=1200 query parameter.
    """
    area = request.args.get("area", None)
    try:
        total_area = float(area) if area is not None else 1000.0
    except ValueError:
        total_area = 1000.0

    return render_template("split.html", total_area=total_area)


# optional UI to preview blueprint (client-side SVG generator preview)
@app.route("/blueprint", methods=["GET"])
def blueprint_page():
    """
    Renders a blueprint preview page (template must exist).
    You can use the standalone blueprint_preview.html if you prefer no server rendering.
    """
    return render_template("blueprint_preview.html")


@app.route("/generate_blueprint", methods=["POST"])
def generate_blueprint():
    """
    Accepts JSON body: { "rooms": [ { "room": "Living Room", "area": 400 }, ... ] }
    Returns: PNG image bytes (download)
    """
    try:
        payload = request.get_json(force=True)
        rooms = payload.get("rooms", [])
        if not rooms or not isinstance(rooms, list):
            return jsonify({"error": "No rooms provided or invalid format"}), 400

        # sanitize & compute total area
        sanitized = []
        total_area = 0.0
        for r in rooms:
            name = str(r.get("room", "Room"))
            try:
                area = float(r.get("area", 0))
            except Exception:
                area = 0.0
            if area > 0:
                sanitized.append({"room": name, "area": area})
                total_area += area

        if total_area <= 0:
            return jsonify({"error": "Total area must be > 0"}), 400

        # Canvas parameters (tweakable)
        img_w, img_h = 1500, 1000
        padding = 16
        bg_color = (15, 15, 15)
        outline_color = (201, 163, 75)  # gold

        img = Image.new("RGB", (img_w, img_h), color=bg_color)
        draw = ImageDraw.Draw(img)

        # fonts: try to load a TTF, otherwise fall back to default
        try:
            # If needed, place a TTF alongside the app and point to it
            font = ImageFont.truetype("arial.ttf", 16)
            font_bold = ImageFont.truetype("arialbd.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
            font_bold = font

        # Simple packing: sort descending and place left-to-right with wrapping
        rooms_sorted = sorted(sanitized, key=lambda r: -r["area"])

        x = padding
        y = padding
        max_row_height = 0
        base_row_height = 200

        # Compute a loose pixels-per-sqft factor to make rooms visually reasonable:
        available_area_pixels = (img_w - padding * 2) * (img_h - padding * 2)
        px_per_sqft = (available_area_pixels * 0.0005) / max(1.0, total_area)

        for r in rooms_sorted:
            name = r["room"]
            area = r["area"]

            # width/height approximation from area and px_per_sqft
            w = max(60, int(area * px_per_sqft * 1.8))
            h = max(60, int(base_row_height * (1.0 + (area / total_area) * 1.4)))

            # wrap to next row if needed
            if x + w + padding > img_w:
                x = padding
                y += max_row_height + padding
                max_row_height = 0

            if y + h + padding > img_h:
                # out of canvas space; stop drawing further rooms
                break

            # draw rectangle with gold outline
            fill_color = (30, 30, 30)
            draw.rectangle([x, y, x + w, y + h], fill=fill_color, outline=outline_color, width=3)

            # draw name and area text
            text = f"{name}\n{int(area)} sq.ft"
            draw.multiline_text((x + 8, y + 8), text, font=font_bold, fill=(255, 255, 255))

            x += w + padding
            max_row_height = max(max_row_height, h)

        # footer
        footer_text = f"Total: {int(total_area)} sq.ft"
        draw.text((padding, img_h - padding - 18), footer_text, font=font, fill=(191, 182, 168))

        # return image as bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        # send as attachment; use download_name for Flask>=2.0
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="blueprint.png")

    except Exception as ex:
        return jsonify({"error": str(ex)}), 500


if __name__ == "__main__":
    # run from project root: python app/app.py
    app.run(debug=True)
