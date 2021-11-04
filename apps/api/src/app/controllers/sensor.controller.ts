import { Body, Controller,Inject,Param, Post } from '@nestjs/common';
import { ClientProxy } from '@nestjs/microservices';
import { timeout } from 'rxjs/operators';

import { SensorRequest } from '@lib/models/sensor';
import { getLogger } from "log4js";

@Controller('connfarm/api/device')
export class SensorController {
  private readonly logger = getLogger();
  constructor(@Inject('REDIS') private redis: ClientProxy) {}

  /**
   * Sensing data from gateway
   */
  @Post(':idx/sensor/:type.json')
  handleSensorData(@Param('idx') idx: string, @Param('type') type: string, @Body() body) {
    const sensorData = new SensorRequest();
    sensorData.idx = idx;
    sensorData.type = type;
    sensorData.data = body[Object.keys(body)[0]];
    this.logger.debug(`${idx} ${type} ${JSON.stringify(body)}`);
    return this.redis.send('sensor', sensorData).pipe(timeout(3000));
  }
}
