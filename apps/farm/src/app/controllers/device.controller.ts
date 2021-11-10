import { Controller, Logger } from "@nestjs/common";
import { Ctx, MessagePattern, Payload, RedisContext } from "@nestjs/microservices";
import { DeviceService } from "../services/device.service";

@Controller()
export class DeviceController {
  private readonly logger = new Logger(DeviceController.name);

  constructor(private readonly deviceService: DeviceService) {}

  @MessagePattern('getSimpleDeviceList')
  getSimpleDeviceList(@Payload() data, @Ctx() context: RedisContext) {
    this.logger.debug(`channel: ${context.getChannel()}, data: ${data}`);
    return this.deviceService.getSimpleDeviceList();
  }
}