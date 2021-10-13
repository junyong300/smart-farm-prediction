import { Expose } from 'class-transformer';

export class InternalDto {
  temp: number;
  wetBulbTemp: number;
  humidity: number;
  co2: number;
  solarRadiation: number;
  sensingTime: Date;
}