import { Module } from '@nestjs/common';
import { DbModule } from '@libs/db';

import { AppController } from './app.controller';
import { AppService } from './app.service';
import { DeviceService } from './services/device.service';
import { DeviceController } from './controllers/device.controller';

@Module({
  imports: [DbModule],
  controllers: [AppController, DeviceController],
  providers: [AppService, DeviceService],
})
export class AppModule {}
