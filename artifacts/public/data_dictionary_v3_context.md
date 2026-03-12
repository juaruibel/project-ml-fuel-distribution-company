# Day04 · Data Dictionary V3_A (raw context)

## Alcance
- Documento generado automáticamente desde el schema final de `dataset_modelo_proveedor_v3_context.csv`.
- Mantiene contrato V2 y añade señales raw/comparativa/dispersión aprobadas para Day 04.

| columna | dtype | rol | justificación |
|---|---|---|---|
| event_id | object | identity | Id único determinista por evento para trazabilidad. |
| fecha_evento | object | context | Fecha operativa del evento de compra. |
| albaran_id | object | audit | Identificador documental del albarán. |
| linea_id | object | audit | Identificador de línea dentro del albarán. |
| producto_canonico | object | context | Producto normalizado para unir compras y ofertas. |
| terminal_compra | object | context | Terminal operativa registrada en compras. |
| proveedor_elegido_real | object | label_source | Proveedor realmente comprado (ground truth). |
| proveedor_candidato | object | candidate_key | Proveedor del universo candidato para el evento. |
| coste_min_dia_proveedor | float64 | feature_base | Coste del candidato en ese día-producto. |
| rank_coste_dia_producto | float64 | feature_base | Posición relativa de coste del candidato. |
| terminales_cubiertos | float64 | feature_base | Cobertura de terminales de la oferta. |
| observaciones_oferta | float64 | feature_base | Número de observaciones agregadas de oferta. |
| candidatos_evento_count | int64 | feature_competition | Número de proveedores candidatos del evento. |
| coste_min_evento | float64 | feature_competition | Coste mínimo entre candidatos del evento. |
| coste_max_evento | float64 | feature_competition | Coste máximo entre candidatos del evento. |
| spread_coste_evento | float64 | feature_competition | Dispersión de costes en el evento. |
| delta_vs_min_evento | float64 | feature_competition | Diferencia absoluta del candidato frente al mínimo. |
| ratio_vs_min_evento | float64 | feature_competition | Diferencia relativa del candidato frente al mínimo. |
| litros_evento | int64 | context | Volumen comprado en la línea. |
| precio_unitario_evento | float64 | audit | Precio real pagado en la compra (auditoría). |
| importe_total_evento | float64 | audit | Importe total real de la línea (auditoría). |
| dia_semana | int64 | feature_calendar | Estacionalidad semanal del evento. |
| mes | int64 | feature_calendar | Estacionalidad mensual del evento. |
| fin_mes | int64 | feature_calendar | Flag de cierre de mes. |
| blocked_by_rule_candidate | int64 | feature_business_rule | Flag de bloqueo por regla de negocio. |
| block_reason_candidate | object | feature_business_rule | Motivo de bloqueo aplicado al candidato. |
| target_elegido | int64 | target | Etiqueta 1/0: candidato elegido vs no elegido. |
| v2_run_id | object | lineage | Identificador de ejecución ETL V2. |
| v2_ts_utc | object | lineage | Timestamp UTC de generación del dataset. |
| v3_selected_source_calculos_share | float64 | feature_quality | Share de registros raw donde la oferta viene seleccionada desde la hoja Cálculos. |
| v3_cost_source_calculos_share | float64 | feature_quality | Share de costes mínimos cuyo origen trazado es Cálculos. |
| v3_reconciliation_conflict_share | float64 | feature_quality | Proporción de observaciones raw con conflicto entre Tabla y Cálculos. |
| v3_reconciliation_single_source_share | float64 | feature_quality | Proporción de observaciones raw donde solo existe una fuente válida. |
| v3_cost_mean_terminal | float64 | feature_dispersion | Coste medio observado entre terminales para ese día-producto-proveedor. |
| v3_cost_std_terminal | float64 | feature_dispersion | Desviación estándar del coste entre terminales del proveedor. |
| v3_cost_range_terminal | float64 | feature_dispersion | Rango de coste entre terminales (`max-min`) para ese proveedor en el día. |
| v3_cost_cv_terminal | float64 | feature_dispersion | Coeficiente de variación del coste entre terminales. |
| v3_share_terminales_min_cost | float64 | feature_dispersion | Share aproximado de terminales que sostienen el mínimo diario del proveedor. |
| v3_coste_segundo_evento | float64 | feature_competition | Segundo mejor coste disponible en el evento. |
| v3_gap_min_vs_second_evento | float64 | feature_competition | Diferencia entre el mejor coste del evento y el segundo mejor. |
| v3_delta_vs_second_evento | float64 | feature_competition | Diferencia del candidato frente al segundo mejor coste del evento. |
| v3_ratio_vs_second_evento | float64 | feature_competition | Ratio del coste del candidato frente al segundo mejor coste del evento. |
| v3_rank_pct_evento | float64 | feature_competition | Posición relativa del candidato dentro del ranking de coste del evento. |
| v3_coste_mean_evento | float64 | feature_competition | Coste medio de todos los candidatos del evento. |
| v3_delta_vs_mean_evento | float64 | feature_competition | Diferencia del candidato frente al coste medio del evento. |
| v3_candidatos_min_coste_count | int64 | feature_competition | Número de candidatos empatados en el coste mínimo del evento. |
| v3_is_unique_min_coste_evento | int64 | feature_competition | Flag 1/0 que indica si existe un único ganador en el coste mínimo del evento. |