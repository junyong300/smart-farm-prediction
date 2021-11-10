import { Device } from '@libs/db/device';
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { SimpleDeviceDto } from '@libs/models/device'

@Injectable()
export class DeviceService {
  constructor(
    @InjectRepository(Device) private deviceRepo: Repository<Device>,
  ) {}

  getSimpleDeviceList(): Promise<SimpleDeviceDto[]> {
    return this.deviceRepo.find({
      select: ['id']
    });
  }
}
