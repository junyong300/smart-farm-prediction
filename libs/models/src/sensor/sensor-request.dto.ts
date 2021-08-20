import { CreateExternalEnvDto } from ".";
import { CreateInternalEnvDto } from "./create-internalEnv.dto";

export class SensorRequest {
  idx: string;
  type: string;
  data: CreateInternalEnvDto | CreateExternalEnvDto;
}