import { Injectable } from '@nestjs/common';
import { Internal } from '@lib/db/sensor';
import { CreateExternalEnvDto, CreateInternalEnvDto, InternalEnvDto, SensorRequest } from '@lib/models/sensor';

@Injectable()
export class AppService {

  getData(): { message: string } {
    return { message: 'Welcome to sensor-collector!' };
  }

  saveSensorData(sensorRequest: SensorRequest) {
    switch(sensorRequest.type) {
      case 'internalEnv': {
        const cie: CreateInternalEnvDto = (sensorRequest.data as CreateInternalEnvDto);
        const internal = new Internal();
        internal.idx = sensorRequest.idx;
        internal.temp = cie.temp;
        internal.sensing_dt = cie.sensingDt;
        internal.save();
        break;
      }
    }
  }
}
