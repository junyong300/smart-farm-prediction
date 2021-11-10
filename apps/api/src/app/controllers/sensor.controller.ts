import { Body, Controller,Inject,Param, Post } from '@nestjs/common';
import { ClientProxy } from '@nestjs/microservices';
import { timeout } from 'rxjs/operators';
import { HttpService } from '@nestjs/axios';

import { SensorRequest } from '@libs/models/sensor';
import { getLogger } from "log4js";
import { ConfigService } from '@nestjs/config';

@Controller('connfarm/api/device')
export class SensorController {
  sendToSensor = false;

  private readonly logger = getLogger();
  constructor(@Inject('REDIS') private redis: ClientProxy, private httpService: HttpService, private config: ConfigService) {
    this.sendToSensor = config.get("API_SEND_SENSOR", false) == 'true';
  }

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

    // for development
    if (this.sendToSensor) {
      this.logger.debug("send copied sensor data to edge")
      const url = `http://192.168.0.188:8080/connfarm/api/device/${idx}/sensor/${type}.json`;
      this.httpService.post(url, body).pipe(timeout(500)).subscribe();
    }

    return this.redis.send('sensor', sensorData).pipe(timeout(3000));
  }
}
