import { Controller, Get } from '@nestjs/common';
import { Ctx, MessagePattern, Payload, RedisContext } from '@nestjs/microservices';
import { SensorRequest } from '@lib/models/sensor';

import { AppService } from './app.service';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @MessagePattern('sensor')
  handleSensorData(@Payload() data: SensorRequest, @Ctx() context: RedisContext) {
    console.log(`channel: ${context.getChannel()}, data: ${data}`);
    this.appService.saveSensorData(data);
    return "ok";
  }

  @MessagePattern('getInternal')
  getInternal(@Payload() data, @Ctx() context: RedisContext) {
    console.log(`channel: ${context.getChannel()}, data: ${data}`);
    return this.appService.getInternal(data);
  }

}
