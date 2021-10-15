create database farmconnect;

CREATE SCHEMA IF NOT EXISTS farm;
CREATE SCHEMA IF NOT EXISTS device;
CREATE SCHEMA IF NOT EXISTS env;
CREATE SCHEMA IF NOT EXISTS common;

CREATE TABLE IF NOT EXISTS Common.Config (
  key varchar(50),
  value varchar(50)
)

CREATE TABLE IF NOT EXISTS Farm.Farm (
  Id smallint,
  Name varchar(50),
  Address varchar(255),
  SidoCd varchar(5),    -- 시도
  SggCd varchar(5),     -- 시군구
  EmdCd varchar(5),     -- 읍면동
  Latitude float,
  Longitude float,
  area float,            -- 면적 m^2
  Equipments varchar(1000),
  CreatedTime timestamp with time zone default now(),
  ModifiedTime timestamp with time zone,
  DeletedTime timestamp with time zone,
  primary key Id
) 

CREATE TABLE IF NOT EXISTS Farm.Dong (
  DongId smallint,
  FarmId smallint,
  IntDevId smallint,
  extDevId smallint,
  CulDevId smallint,
  NutDevId smallint,
  SoilDevId smallint,
  isMain  boolean,
  CreatedTime timestamp with time zone default now(),
  ModifiedTime timestamp with time zone,
  DeletedTime timestamp with time zone
)

CREATE TABLE IF NOT EXISTS Device.Device (
  Id integer, -- (10~99)(0~9)(0000~9999)
  FarmId integer,
  DongId smallint,
  CommType smallint default 1,
  ModelId smallint,
  Serial varchar(50),
  FormatId smallint,
  SensingInterval smallint default 1,
  State smallint default 0,
  FirstActiveTime timestamp with time zone,
  LastActiveTime timestamp with time zone,
  CreatedTime timestamp with time zone default now(),
  ModifiedTime timestamp with time zone,
  DeletedTime timestamp with time zone,
  primary key (Id)
)
-- COMMENT ON COLUMN Device.Device.Id IS 'Device Id';
-- COMMENT ON COLUMN Device.Device.DongId IS 'Dong Id on Farm.Dong table';

CREATE TABLE IF NOT EXISTS Env.External (
  DeviceId integer,
  SensingTime timestamp with time zone, -- cut seconds
  Temp float,
  Humidity float,
  SolarRadiation float,
  WindSpeed float,
  WindDirection float,
  Rainfall float,
  primary key (DeviceId, SensingTime)
)

CREATE TABLE IF NOT EXISTS Env.Internal (
  DeviceId integer,
  SensingTime timestamp with time zone, -- cut seconds
  Temp float,
  Humidity float,
  WetBulbTemp float,
  CO2 float,
  DewPointTemp float,
  HD float,
  SolarRadiation float,
  --RootTemp float,
  --RootHumidity float,
  primary key (DeviceId, SensingTime)
)
