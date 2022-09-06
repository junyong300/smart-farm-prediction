import { Module } from '@nestjs/common';
import { Device, TENANT, TenantModule } from '@libs/db';

import { AppController } from './app.controller';
import { AppService } from './app.service';
import { DeviceService } from './services/device.service';
import { DeviceController } from './controllers/device.controller';
import { TypeOrmModule } from '@nestjs/typeorm';
import { CommModule } from '@libs/comm';

@Module({
  imports: [CommModule, TenantModule, TypeOrmModule.forFeature([Device], TENANT)],
  controllers: [AppController, DeviceController],
  providers: [AppService, DeviceService],
})
export class AppModule {}
