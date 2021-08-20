import { Expose } from 'class-transformer';

export class InternalEnvDto {
  temp: number;
  dryBulbTemp: number;
  wetBulbTemp: number;
  humidity: number;
  co2: number;
  lightness: number;
  solarRadiation: number;
  illuminance: number;
  sensingDt: Date;
}