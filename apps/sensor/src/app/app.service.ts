import { Injectable } from '@nestjs/common';
import { Internal } from '@lib/db/env';
import { Device } from '@lib/db/device';
import { CreateInternalEnvDto, SensorRequest } from '@lib/models/sensor';
import { plainToClass } from 'class-transformer';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';

@Injectable()
export class AppService {
  constructor(
    @InjectRepository(Device) private deviceRepo: Repository<Device>,
    @InjectRepository(Internal) private internalRepo: Repository<Internal>,
    ) {}

  async saveSensorData(sensorRequest: SensorRequest) {
    switch(sensorRequest.type) {
      case 'internalEnv': {
        const device = await Device.findOne({ where: { serial: sensorRequest.idx }});
        if (!device) {
          console.log("No device");
          return;
        }
        const deviceId = device.id;

        const cie = plainToClass(CreateInternalEnvDto, sensorRequest.data);

        const internal = new Internal();
        internal.deviceId = deviceId;
        internal.sensingTime = cie.sensingDt;
        internal.sensingTime.setSeconds(0, 0);
        internal.temp = cie.temp;
        internal.humidity = cie.temp;
        internal.wetBulbTemp = cie.wetBulbTemp;
        internal.co2 = cie.co2;
        internal.solarRadiation = cie.solarRadiation;

        this.internalRepo.insert(internal);
        this.updateFirstLastActiveTime(device, new Date(sensorRequest.data.sensingDt));

        break;
      }
    }
  }

  /**
   * Update with including seconds
   * @param device 
   * @param sensingTime 
   */
  updateFirstLastActiveTime(device: Device, sensingTime: Date) {
    if (!device.firstActiveTime || device.lastActiveTime < sensingTime) {
      if (!device.firstActiveTime) {
        device.firstActiveTime = sensingTime;
      }
      device.lastActiveTime = sensingTime;
      this.deviceRepo.update(
        { id: device.id },
        { firstActiveTime: device.firstActiveTime, lastActiveTime: device.lastActiveTime }
      );
    }
  }

  getInternal(param) {
    return this.internalRepo.find({
      order: {
        sensingTime: 'DESC'
      },
      take: 100
    });
  }
}
