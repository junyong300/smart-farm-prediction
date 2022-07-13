export class PredictEnvOptionsDto {
  inDeviceIdx: number;
  exDeviceIdx: number;
  time?: string; // 'yyyy-mm-dd hh:mm:ss'
}
export class PredictInputDto {
  modelId: string;
  params: any;
  data: any;
}

export class PredictOptionsDto {
  modelId?: string;
  deviceIdx?: number;
  time?: string; // 'yyyy-mm-dd hh:mm:ss'
}

export class PredictOutputDto {

}