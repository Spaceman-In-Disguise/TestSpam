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
        body {
            font-family: sans-serif;
            max-width: 800px;
            margin: 2rem auto;
            padding: 0 1rem;
        }
        textarea {
            width: 100%;
            min-height: 160px;
            margin-bottom: 1rem;
            padding: 0.75rem;
            font-size: 1rem;
        }
        button {
            padding: 0.7rem 1rem;
            font-size: 1rem;
            cursor: pointer;
        }
        .result {
            margin-top: 1rem;
            font-size: 1.1rem;
            font-weight: 600;
        }
        .error {
            margin-top: 1rem;
            color: #b00020;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <h1>Spam Detector</h1>
    <form method="post">
        <label for="texto">Enter text:</label><br>
        <textarea id="texto" name="texto" placeholder="Write the email/message here...">{{ texto }}</textarea>
        <br>
        <button type="submit">Classify</button>
    </form>
    {% if result %}
        <div class="result">Result: {{ result }}</div>
    {% endif %}
    {% if error %}
        <div class="error">{{ error }}</div>
    {% endif %}
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
    app.run(debug=True)