# Daily Prompt History

Histórico oficial de prompts de sesión por día para mantener contexto limpio y trazable.

## Convención de nombres

- Base diaria: `<YYYYMMDD>_dayNN_prompt.md`
- Múltiples sesiones en el mismo día: `<YYYYMMDD>_dayNN_runNN_prompt.md`

## Contenido mínimo obligatorio

1. Metadata:
- `date_utc`
- `day_id`
- `base_prompt_ref`
2. Prompt operativo del día.
3. `Delta vs día anterior` con cambios de alcance/contexto.
4. Referencia al notebook diario donde se documentan decisiones y razonamientos.

## Regla operativa

- Cada cierre de día crea o actualiza un archivo en esta carpeta.
- Cada cambio de archivo diario debe reflejarse en `src/prompts/prompt_artifact_manifest.yaml`.
- Cada día debe abrir un notebook nuevo en `notebooks/` (serie 05+) para dejar constancia técnica y de negocio.
