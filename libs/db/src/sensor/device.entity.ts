import { BeforeInsert, Column, Entity, PrimaryGeneratedColumn } from 'typeorm';

export enum CommType {
  DEFAULT = 1,
  REST_API = 2,
  GREENCS = 3,
  WOOSUNG = 4,
  NMS = 5,
  KMA = 6
};


@Entity({ name: 'device', synchronize: false })
export class Device {
  @PrimaryGeneratedColumn({ name: 'DEVICE_IDX', comment: '' })
  idx?: number;

  @Column({ name: 'DELETE_DT', comment: '' })
  deleteDt?: Date;

  @Column({ name: 'DEVICE_SERIAL_NUMBER', comment: '' })
  serialNumber?: string;

  @Column({ name: 'DEVICE_COMM_CD', comment: ''})
  commType?: number;
  @Column({ name: 'DEVICE_COMM_VALUE', comment: '' })
  commValue?: string;

  @Column({ name: 'USE_FLAG', comment: '' })
  use?: boolean;
  @Column({ name: 'USE_REG_DT', comment: '' })
  useRegDt?: Date;
  @Column({ name: 'USE_USER_IDX', comment: '' })
  useUserIdx?: number;
  @Column({ name: 'USE_FARM_DONG_IDX', comment: '' })
  useFarmDongIdx?: number;

  @Column({ name: 'MODEL_CD', comment: '' })
  modelCd?: number = 1;

  @Column({ name: 'FORMAT_IDX', comment: ''})
  formatIdx?: number = 0;

  @Column({ name: 'DEVICE_REPORT_PERIOD', comment: ''})
  reportPeriod?: number;
  @Column({ name: 'DEVICE_FIRST_REPORT_DT', comment: '' })
  firstReportDt?: Date;
  @Column({ name: 'DEVICE_LAST_REPORT_DT', comment: '' })
  lastReportDt?: Date;

  @Column({ name: 'DEVICE_COMM_ERROR_FLAG', comment: '' })
  commError?: boolean = false;

  @Column({ name: 'CREATE_DT', comment: '' })
  createDt?: Date;
  @Column({ name: 'MODIFY_DT', comment: '' })
  updateDt?: Date;

  @BeforeInsert()
  beforeInsert() {
    this.createDt = new Date();
    this.updateDt = new Date();
  }
}
