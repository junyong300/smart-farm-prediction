import { Type, Transform } from "class-transformer";
export class CreateInternalEnvDto {
  @Type(() => Date)
  sensingDt: Date;
  temp: number;
  dryBulbTemp: number;
  wetBulbTemp: number;
  humidity: number;
  co2: number;
  lightness: number;
  solarRadiation: number;
  illuminance: number;
}