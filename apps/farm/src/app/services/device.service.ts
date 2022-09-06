import { Device } from '@libs/db/sensor';
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { SimpleDeviceDto } from '@libs/models/device'
import { TENANT } from '@libs/db';

@Injectable()
export class DeviceService {
  constructor(
    @InjectRepository(Device, TENANT) private deviceRepo: Repository<Device>,
  ) {}

  /*
  getSimpleDeviceList(): Promise<SimpleDeviceDto[]> {
    return this.deviceRepo.find({
      select: ['id']
    });
  }
  */

  deviceList() {
    return this.deviceRepo.find();
  }
}
