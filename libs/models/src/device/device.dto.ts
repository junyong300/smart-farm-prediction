import { Expose } from "class-transformer";

export class DeviceDto {
  @Expose()
  idx: number;
  @Expose()
  serial: string;
}