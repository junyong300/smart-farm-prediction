import { Device } from "@libs/db";
import { DeviceDto } from "@libs/models/device";
import { Controller, Logger } from "@nestjs/common";
import { Ctx, MessagePattern, Payload, RedisContext } from "@nestjs/microservices";
import { plainToInstance } from 'class-transformer';
import { DeviceService } from "../services/device.service";

@Controller()
export class DeviceController {
  private readonly logger = new Logger(DeviceController.name);

  constructor(private readonly deviceService: DeviceService) {}

  @MessagePattern('device.list')
  async deviceList(@Payload() data, @Ctx() context: RedisContext) {
    this.logger.debug(`channel: ${context.getChannel()}, data: ${JSON.stringify(data)}`);
    const devices = await this.deviceService.deviceList();
    return devices.map(d => plainToInstance(DeviceDto, d, {excludeExtraneousValues: true}));
  }
}