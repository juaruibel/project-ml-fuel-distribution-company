from __future__ import annotations

from typing import Any


DEMO_OVERVIEW_BULLETS = [
    "El problema operativo no es predecir por deporte: es reducir tiempo de decision sin perder control humano.",
    "Day 06 junta producto, modelo y trazabilidad en una sola app presentable.",
    "La demo ensena tres cosas: por que el champion es defendible, que se intento en slices dificiles y como se usa la app en la practica.",
]


NOTEBOOK_EXPANDERS: list[dict[str, Any]] = [
    {
        "title": "Notebook 19 · Error analysis y SHAP",
        "intro": (
            "Este bloque explica por que el champion puro es defendible: se reconcilia contra el baseline, "
            "se miran errores reales y se usan slices + SHAP para entender donde el modelo aporta y donde aun necesita supervision."
        ),
        "cards": [
            {
                "title": "Reconciliacion oficial",
                "body": "El champion mejora al baseline historico en Top-1, Top-2 y balanced accuracy, asi que no es solo un cambio cosmetico.",
            },
            {
                "title": "Slices operativas",
                "body": "El analisis separa PRODUCT_005, PRODUCT_003 dominante, PRODUCT_003 residual y la bolsa both_fail_top2 para no mezclar errores de naturaleza distinta.",
            },
            {
                "title": "Lectura SHAP util",
                "body": "SHAP no se usa para decorar: se usa para explicar que senales empujan una decision y para detectar si hay una regla local que merezca investigarse.",
            },
            {
                "title": "Conclusión defendible",
                "body": "La conclusion no es que el modelo sea perfecto, sino que el champion es mejor referencia operativa que el baseline y sabemos ya donde sigue fallando.",
            },
        ],
        "slides": [
            {
                "filename": "nb19_reconciliacion_champion_vs_baseline.png",
                "title": "Slide · Champion vs baseline",
                "prompt_text": (
                    "Explica en una sola slide que el champion puro Day 05.1 se reconcilia contra el baseline historico "
                    "y mejora las metricas operativas clave. El mensaje principal es: no hemos cambiado el modelo porque si, "
                    "sino porque mejora Top-1, Top-2 y balanced accuracy frente al serving historico congelado."
                ),
            },
            {
                "filename": "nb19_error_analysis_y_slices.png",
                "title": "Slide · Error analysis por slices",
                "prompt_text": (
                    "Explica que el error analysis separa los casos en slices operativas: PRODUCT_005, PRODUCT_003 dominante, PRODUCT_003 residual "
                    "y both_fail_top2. El mensaje principal es que no todos los errores significan lo mismo y que esa separacion "
                    "permite decidir si conviene una regla local, una flag de revision o simplemente aceptar que falta senal."
                ),
            },
            {
                "filename": "nb19_shap_como_defensa_del_modelo.png",
                "title": "Slide · SHAP como defensa del champion",
                "prompt_text": (
                    "Explica que SHAP se usa aqui como herramienta de interpretabilidad practica: ayuda a entender que variables "
                    "empujan la recomendacion del champion y por que el modelo es defendible en presentacion. El mensaje principal "
                    "es que SHAP no sustituye a negocio, pero evita vender una caja negra opaca."
                ),
            },
        ],
    },
    {
        "title": "Notebook 20 · Fallback flags y policies locales",
        "intro": (
            "Este bloque es la auditoria tecnica del dia 05.4: que policies locales se probaron, que evidencia hubo que invalidar, "
            "que reruns quedaron como validos y por que ninguna policy nueva se promovio al producto."
        ),
        "cards": [
            {
                "title": "Auditoria, no humo",
                "body": "Se deja separada la evidencia invalidada de la evidencia valida para no mezclar conclusiones no reproducibles con reruns auditados.",
            },
            {
                "title": "Mejoras locales reales",
                "body": "Hubo slices donde una flag o fallback local si parecia prometedora, especialmente en PRODUCT_005 y parte del residual de PRODUCT_003.",
            },
            {
                "title": "Gate por encima del entusiasmo",
                "body": "Que una policy mejore un subconjunto no basta: para promocionar tiene que superar el gate operativo completo sin danar overrides ni coherencia.",
            },
            {
                "title": "Resultado honesto",
                "body": "El bloque 05.4 aporta aprendizaje accionable, pero no justifica una nueva policy operativa por defecto.",
            },
        ],
        "slides": [
            {
                "filename": "nb20_auditoria_de_policies_locales.png",
                "title": "Slide · Que se probo en policies locales",
                "prompt_text": (
                    "Explica que en Notebook 20 se probaron fallback flags y policies locales sobre slices concretas, "
                    "no un cambio global del producto. El mensaje principal es que se investigo donde el champion aun fallaba "
                    "para ver si habia reglas locales defendibles."
                ),
            },
            {
                "filename": "nb20_evidencia_invalida_vs_valida.png",
                "title": "Slide · Evidencia invalida vs valida",
                "prompt_text": (
                    "Explica que una parte de la evidencia inicial del day 05.4 quedo invalidada por un scoring contract mismatch "
                    "y que luego se rehizo con reruns auditados. El mensaje principal es que la metodologia se corrigio antes de sacar conclusiones."
                ),
            },
            {
                "filename": "nb20_por_que_no_se_promovio.png",
                "title": "Slide · Por que no se promovio una policy nueva",
                "prompt_text": (
                    "Explica que algunas policies mejoraron slices concretas, pero ninguna supero el gate completo frente a la mejor policy operativa vigente. "
                    "El mensaje principal es que se prefirio una conclusion honesta a sobreprometer una mejora no suficientemente robusta."
                ),
            },
        ],
    },
    {
        "title": "Notebook 21 · Cierre operativo y resumen para presentacion",
        "intro": (
            "Este bloque resume el valor real de Day 05.3 y Day 05.4 para la presentacion final: que si se gano, "
            "que sigue igual y por que el producto queda defendible aunque no todo se haya promocionado."
        ),
        "cards": [
            {
                "title": "Que si quedo",
                "body": "Quedo un champion puro defendible, una lectura mas clara de los slices dificiles y una auditoria tecnica trazable de las microiteraciones locales.",
            },
            {
                "title": "Que no se promociono",
                "body": "No se promocionaron las nuevas policies Day 05.4 porque el cierre auditado reafirma mantener la mejor policy operativa vigente.",
            },
            {
                "title": "Mensaje de producto",
                "body": "La app no vende magia: vende un sistema asistido que propone, deja trazabilidad y puede explicarse delante de negocio.",
            },
            {
                "title": "Cierre ejecutivo",
                "body": "El proyecto llega a Day 06 con suficiente solidez para presentar valor, metodo y limites sin necesidad de abrir otra ronda de modelado.",
            },
        ],
        "slides": [
            {
                "filename": "nb21_que_aprendimos.png",
                "title": "Slide · Que aprendimos del bloque",
                "prompt_text": (
                    "Explica que Notebook 21 cierra el bloque con tres ideas: el champion puro queda defendible, "
                    "los slices dificiles ya estan identificados y el trabajo de fallback local deja aprendizaje aunque no cambie el producto por defecto."
                ),
            },
            {
                "filename": "nb21_que_si_quedo_y_que_no.png",
                "title": "Slide · Que si quedo y que no",
                "prompt_text": (
                    "Explica con honestidad que si quedo un champion puro defendible y una auditoria tecnica clara, "
                    "pero no se promovio una nueva policy operativa desde Day 05.4. El mensaje principal es que el proyecto avanza sin vender mejoras no cerradas."
                ),
            },
        ],
    },
]


HOW_TO_USE_EXPANDER: dict[str, Any] = {
    "title": "Cómo usar la app",
    "intro": (
        "Este bloque sirve para la demo en vivo y para tus companeros: ensena el recorrido minimo para probar la app "
        "sin entrar en scoring interno, SQL ni detalles tecnicos."
    ),
    "cards": [
        {
            "title": "Paso 1",
            "body": "Cargar la comparativa del dia o usar el ejemplo rapido.",
        },
        {
            "title": "Paso 2",
            "body": "Completar albaran y litros del pedido hasta dejarlo listo para calcular.",
        },
        {
            "title": "Paso 3",
            "body": "Obtener la propuesta champion y revisar solo la informacion necesaria para decidir.",
        },
        {
            "title": "Paso 4",
            "body": "Guardar la decision final para dejar trazabilidad operativa.",
        },
    ],
    "slides": [
        {
            "filename": "how_to_use_recorrido_4_pasos.png",
            "title": "Slide · Recorrido de uso",
            "prompt_text": (
                "Explica el uso de la app en cuatro pasos claros: cargar comparativa, completar pedido, obtener propuesta y revisar/guardar. "
                "El mensaje principal es que el producto esta pensado para reducir friccion diaria, no para sustituir al decisor humano."
            ),
        },
        {
            "filename": "how_to_use_demo_60_segundos.png",
            "title": "Slide · Demo en 60 segundos",
            "prompt_text": (
                "Explica como ensenar la app en menos de un minuto: abrir Producto-Usuario, cargar ejemplo, rellenar pedido, "
                "calcular propuesta y mostrar el guardado final. El mensaje principal es que la demo se entiende sin explicar scoring interno."
            ),
        },
    ],
}
