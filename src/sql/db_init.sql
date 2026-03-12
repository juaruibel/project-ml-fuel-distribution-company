-- db_init.sql
-- Inicialización idempotente de MySQL para capa SQL de serving

CREATE DATABASE IF NOT EXISTS proyecto_ml_serving
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

USE proyecto_ml_serving;

-- Opcional pero recomendado: tabla mínima de auditoría de runs SQL
CREATE TABLE IF NOT EXISTS sql_serving_runs (
  run_id            VARCHAR(32) PRIMARY KEY,
  executed_at_utc   DATETIME NOT NULL,
  executed_by       VARCHAR(128) NULL,
  notes             VARCHAR(255) NULL
);
