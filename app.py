import os
from pathlib import Path
import re

import joblib
import numpy as np
from flask import Flask, jsonify, render_template_string, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


stop_words = set(ENGLISH_STOP_WORDS)


def obtener_promedio_embeddings(tokens_list, modelo, tamano_vector):
    """Calcula el promedio de embeddings de una lista de tokens."""
    vectores = [modelo.wv[word] for word in tokens_list if word in modelo.wv]

    if len(vectores) == 0:
        return np.zeros(tamano_vector)

    return np.mean(vectores, axis=0)


def limpiar_y_tokenizar(texto):
    """Limpia y tokeniza el texto siguiendo la misma logica del entrenamiento."""
    texto = str(texto).lower()

    texto = re.sub(r"\n", " ", texto)
    texto = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", " mailid ", texto)
    texto = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " links ",
        texto,
    )
    texto = re.sub(r"\u00a3|\$|\u20ac", " money ", texto)
    texto = re.sub(
        r"\b(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)[-.\s]?\d{3}[-.\s]?\d{4})\b",
        " contactnumber ",
        texto,
    )
    texto = re.sub(r"[^a-z\s]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    tokens = texto.split()
    return [word for word in tokens if word not in stop_words]


def clasificar_nuevo_correo(texto_correo, modelo_w2v, modelo_clasificacion):
    """Recibe texto crudo y devuelve True si es spam, False en caso contrario."""
    if not isinstance(texto_correo, str):
        raise ValueError("El texto a clasificar debe ser string.")
    if not texto_correo.strip():
        raise ValueError("El texto a clasificar no puede estar vacio.")
    if not hasattr(modelo_w2v, "wv"):
        raise ValueError("El modelo Word2Vec no es valido.")
    if not hasattr(modelo_clasificacion, "predict"):
        raise ValueError("El modelo de clasificacion no es valido.")

    tokens = limpiar_y_tokenizar(texto_correo)
    vector_size = int(getattr(modelo_w2v, "vector_size", 100))
    vector_correo = obtener_promedio_embeddings(tokens, modelo_w2v, vector_size)
    vector_correo_2d = vector_correo.reshape(1, -1)

    prediccion = modelo_clasificacion.predict(vector_correo_2d)[0]
    return bool(prediccion == 1)


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
W2V_MODEL_PATH = MODEL_DIR / "modelo_w2v.joblib"
WINNER_MODEL_PATH = MODEL_DIR / "modelo_ganador.joblib"

if not W2V_MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontro el modelo Word2Vec: {W2V_MODEL_PATH}")
if not WINNER_MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontro el modelo clasificador: {WINNER_MODEL_PATH}")

modelo_w2v = joblib.load(W2V_MODEL_PATH)
modelo_ganador = joblib.load(WINNER_MODEL_PATH)

app = Flask(__name__)


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Spam Detector</title>
    <style>
        :root {
            --bg-0: #11081d;
            --bg-1: #1a1030;
            --bg-2: #31165e;
            --card: rgba(28, 17, 48, 0.86);
            --card-border: #55318a;
            --text: #f2e8ff;
            --muted: #cdbde8;
            --input-bg: #1d1333;
            --input-border: #66439c;
            --accent-1: #8d5dff;
            --accent-2: #c064ff;
            --error: #ff7eaf;
            --spam: #ff9ecf;
            --ham: #7af2cb;
        }

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            padding: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            color: var(--text);
            background:
                radial-gradient(circle at 15% 15%, #3a1f6b 0%, transparent 40%),
                radial-gradient(circle at 85% 85%, #4e2385 0%, transparent 38%),
                linear-gradient(155deg, var(--bg-0), var(--bg-1) 45%, var(--bg-2));
        }

        .app-card {
            width: min(820px, 100%);
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 26px;
            box-shadow: 0 24px 50px rgba(6, 3, 12, 0.55);
            backdrop-filter: blur(3px);
        }

        h1 {
            margin: 0;
            font-size: 1.9rem;
            letter-spacing: 0.3px;
        }

        .subtitle {
            margin: 8px 0 18px;
            color: var(--muted);
            font-size: 0.98rem;
        }

        label {
            display: block;
            margin-bottom: 0.45rem;
            font-size: 0.95rem;
            color: var(--muted);
        }

        textarea {
            width: 100%;
            min-height: 170px;
            margin-bottom: 0.9rem;
            padding: 0.9rem;
            border-radius: 12px;
            border: 1px solid var(--input-border);
            background: var(--input-bg);
            color: var(--text);
            font-size: 1rem;
            line-height: 1.5;
            outline: none;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }

        textarea:focus {
            border-color: var(--accent-2);
            box-shadow: 0 0 0 3px rgba(192, 100, 255, 0.2);
        }

        button {
            border: none;
            border-radius: 11px;
            padding: 0.78rem 1.15rem;
            font-size: 0.98rem;
            font-weight: 600;
            color: #f9f5ff;
            background: linear-gradient(120deg, var(--accent-1), var(--accent-2));
            cursor: pointer;
            transition: transform 0.12s ease, filter 0.18s ease;
        }

        button:hover {
            filter: brightness(1.08);
        }

        button:active {
            transform: translateY(1px);
        }

        .result,
        .error {
            margin-top: 1rem;
            padding: 0.72rem 0.85rem;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
        }

        .result {
            border: 1px solid #7041b0;
            background: rgba(95, 59, 146, 0.26);
        }

        .result.spam {
            color: var(--spam);
        }

        .result.ham {
            color: var(--ham);
        }

        .error {
            color: var(--error);
            border: 1px solid #95407b;
            background: rgba(140, 45, 96, 0.25);
        }

        @media (max-width: 640px) {
            body {
                padding: 14px;
            }

            .app-card {
                padding: 18px;
                border-radius: 14px;
            }

            h1 {
                font-size: 1.55rem;
            }
        }
    </style>
</head>
<body>
    <main class="app-card">
        <h1>Spam Detector</h1>
        <p class="subtitle">Simple spam check powered by your trained models.</p>

        <form method="post">
            <label for="texto">Message or email text</label>
            <textarea id="texto" name="texto" placeholder="Write the email/message here...">{{ texto }}</textarea>
            <button type="submit">Classify</button>
        </form>

        {% if result %}
            <div class="result {{ 'spam' if result == 'Spam' else 'ham' }}">Result: {{ result }}</div>
        {% endif %}
        {% if error %}
            <div class="error">{{ error }}</div>
        {% endif %}
    </main>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    texto = ""

    if request.method == "POST":
        texto = request.form.get("texto", "")

        if not texto.strip():
            error = "Please enter a text to classify."
        else:
            try:
                es_spam = clasificar_nuevo_correo(texto, modelo_w2v, modelo_ganador)
                result = "Spam" if es_spam else "Not Spam"
            except ValueError as exc:
                error = str(exc)
            except Exception:
                error = "Unexpected error while classifying text."

    return render_template_string(HTML_TEMPLATE, texto=texto, result=result, error=error)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    texto = data.get("text", "")

    if not isinstance(texto, str) or not texto.strip():
        return (
            jsonify({"error": "The 'text' field is required and must be a non-empty string."}),
            400,
        )

    try:
        es_spam = clasificar_nuevo_correo(texto, modelo_w2v, modelo_ganador)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Could not classify text: {exc}"}), 500

    return jsonify(
        {
            "text": texto,
            "is_spam": bool(es_spam),
            "label": "spam" if es_spam else "not_spam",
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)