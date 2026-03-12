-- Crea la view para serving

CREATE OR REPLACE VIEW vw_inference_input_daily AS
SELECT
  event_id, fecha_evento, proveedor_candidato, producto_canonico, terminal_compra,
  coste_min_dia_proveedor, rank_coste_dia_producto, terminales_cubiertos, observaciones_oferta,
  candidatos_evento_count, coste_min_evento, coste_max_evento, spread_coste_evento,
  delta_vs_min_evento, ratio_vs_min_evento, litros_evento, dia_semana, mes, fin_mes
FROM marts_dataset_modelo_proveedor_v2_candidates;

CREATE OR REPLACE VIEW vw_event_summary_daily AS
SELECT
  event_id,
  MAX(candidatos_evento_count) AS candidatos_evento_count,
  MIN(coste_min_evento) AS coste_min_evento,
  MAX(coste_max_evento) AS coste_max_evento,
  MAX(coste_max_evento) - MIN(coste_min_evento) AS spread_coste_evento
FROM vw_inference_input_daily
GROUP BY event_id;
