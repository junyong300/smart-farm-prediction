CREATE TABLE IF NOT EXISTS sensor.internal (
  idx char(10)  NOT NULL,
  sensing_dt TIMESTAMPZ NOT NULL,
  create_dt TIMESTAMPZ NOT NULL,
  temp FLOAT,
  PRIMARY KEY (idx, sensing_date)
)