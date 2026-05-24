from flask import Flask
from flask_restx import Api, Resource, fields
import pandas as pd

app = Flask(__name__)

api = Api(
    app,
    version="1.0",
    title="API de Predicción de Géneros",
    description="API para consultar probabilidades de género por ID"
)

ns = api.namespace("predicciones", description="Consulta de predicciones")

# Cargar predicciones una sola vez al iniciar la API
df_pred = pd.read_csv("predicciones_model6_ensemble_optimo.csv")

# Asegurar que ID sea string para evitar problemas
df_pred["ID"] = df_pred["ID"].astype(str)

# Modelo de salida para Swagger
genero_model = api.model("GeneroPrediccion", {
    "ID": fields.String(description="ID de la película"),
    "Genero": fields.String(description="Género"),
    "Prediccion": fields.Float(description="Probabilidad predicha")
})


@ns.route("/<string:id>")
@ns.param("id", "ID de la película a consultar")
class PrediccionPorID(Resource):

    @ns.marshal_list_with(genero_model)
    def get(self, id):
        """
        Retorna las probabilidades de todos los géneros para un ID específico.
        """

        fila = df_pred[df_pred["ID"] == id]

        if fila.empty:
            api.abort(404, f"No se encontró el ID {id}")

        fila = fila.iloc[0]

        resultado = []

        for col in df_pred.columns:
            if col != "ID":
                genero = col.replace("p_", "")

                resultado.append({
                    "ID": id,
                    "Genero": genero,
                    "Prediccion": float(fila[col])
                })

        return resultado


@ns.route("/")
class TodasLasPredicciones(Resource):

    def get(self):
        """
        Retorna mensaje básico de uso.
        """
        return {
            "mensaje": "Use /predicciones/<ID> para consultar las probabilidades por género."
        }


if __name__ == "__main__":
    app.run(debug=False)