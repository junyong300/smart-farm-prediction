import { BeforeInsert, Column, Entity, PrimaryGeneratedColumn } from 'typeorm';

@Entity({ name: 'farm_dong' })
export class FarmDong {
  @PrimaryGeneratedColumn({ name: 'FARM_DONG_IDX', comment: 'serial' })
  idx?: number;

  @Column({ name: 'FARM_IDX', comment: '' })
  farmIdx?: number;
  @Column({ name: 'FARM_DONG_NO', comment: '' })
  no?: number;
  @Column({ name: 'MAIN_DONG_FLAG', comment: '' })
  mainDong?: boolean;
  @Column({ name: 'ACTIVE', comment: '' })
  active?: boolean;

  @Column({ name: 'IN_ENV_DEVICE_IDX', comment: '' })
  inEnvDeviceIdx?: number;
  @Column({ name: 'EXT_WEATHER_DEVICE_IDX', comment: ''})
  extWeatherDeviceIdx?: number;
  @Column({ name: 'NUTRIENT_SOLUTION_DEVICE_IDX', comment: '' })
  nutrientSolutionDeviceIdx?: number;
  @Column({ name: 'CULTURE_MEDIUM_DEVICE_IDX', comment: '' })
  cultureMediumDeviceIdx?: number;
  @Column({ name: 'SOIL_DEVICE_IDX', comment: '' })
  soilDeviceIdx?: number;

  @Column({ name: 'CREATE_DT', comment: '' })
  createDt?: Date;
  @Column({ name: 'MODIFY_DT', comment: '' })
  updateDt?: Date;
  @Column({ name: 'DELETE_DT', comment: '' })
  deleteDt?: Date;

  @BeforeInsert()
  beforeInsert() {
    this.createDt = new Date();
    this.updateDt = new Date();
  }

}
