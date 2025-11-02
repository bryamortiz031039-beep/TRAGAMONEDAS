# InplayGuru Live Pressure Analytics

Este repositorio contiene un conjunto de utilidades para descargar métricas en vivo de
[InplayGuru](https://inplayguru.com), transformar dichas métricas en indicadores de
presión ofensiva y estimar la probabilidad de que ocurra un gol en el corto plazo.

## Componentes principales

- `inplayguru_analytics/clients/inplayguru_client.py`: cliente HTTP basado en
  `requests` que encapsula la comunicación con el backend de InplayGuru. Incluye
  utilidades para normalizar los payloads JSON.
- `inplayguru_analytics/features/pressure_features.py`: funciones de ingeniería de
  características que suavizan y combinan las métricas en crudos (posesión, ataques
  peligrosos, xG, etc.) para generar una señal de presión.
- `inplayguru_analytics/models/pressure_model.py`: implementación ligera de un modelo
  de regresión logística que puede entrenarse de forma batch u online.
- `inplayguru_analytics/streaming/monitor.py`: bucle de monitoreo que integra el
  cliente, las características y el modelo para producir predicciones en tiempo real.
- `scripts/run_live_monitor.py`: script CLI que demuestra el flujo completo usando un
  archivo JSON con snapshots históricos.

## Requisitos

1. Python 3.10+
2. Dependencias de runtime:
   ```bash
   pip install requests
   ```

El modelo y las funciones de ingeniería no dependen de bibliotecas externas pesadas
como `scikit-learn`. Si se desea reemplazar el modelo por uno más sofisticado se puede
hacer reutilizando los mismos contratos.

## Ejecución de ejemplo con datos offline

1. Entrenar un modelo stub y generar predicciones usando el dataset de ejemplo:

   ```bash
   python scripts/run_live_monitor.py DEMO_MATCH --history sample_data/example_metrics.json --stop-after 5
   ```

2. El script precargará el historial, entrenará un modelo de demostración y mostrará la
   probabilidad estimada de que ocurra un gol durante las próximas posesiones.

## Integración con métricas en vivo

Para operar contra partidos en vivo se necesita una sesión válida de InplayGuru. Una
vez obtenidas las cookies (por ejemplo, desde el navegador), se pueden inyectar así:

```python
from inplayguru_analytics.clients.inplayguru_client import InplayGuruClient, InplayGuruClientConfig
from inplayguru_analytics.streaming.monitor import LivePressureMonitor
from inplayguru_analytics.models.pressure_model import PressureGoalModel

config = InplayGuruClientConfig(
    headers={"User-Agent": "my-research-bot/1.0"},
    cookies={"session": "<tu_cookie_de_sesion>"},
)
client = InplayGuruClient(config)
model = PressureGoalModel()
monitor = LivePressureMonitor(client=client, model=model)
monitor.run(match_id="12345")
```

El callback `on_prediction` permite conectar las salidas del modelo con sistemas de
alertas, dashboards o estrategias de trading automático.
