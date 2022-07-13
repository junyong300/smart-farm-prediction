import { CreateSensorDataOldDto } from "./create-sensor-data-old.dto";
import { CreateSensorDataDto } from "./create-sensor-data.dto";

/**
 * for queue data
 */
export class SensorJobDataDto {
  originalUrl: string;
  // headers: {[key: string]: string};
  headers: NodeJS.Dict<string|string[]>;
  body: CreateSensorDataOldDto | CreateSensorDataDto | CreateSensorDataDto[];
}