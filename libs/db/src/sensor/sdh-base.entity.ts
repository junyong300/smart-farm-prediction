import { PrimaryGeneratedColumn, Column, BeforeInsert, CreateDateColumn } from 'typeorm';

export abstract class SdhBase {
  @PrimaryGeneratedColumn({ name: 'IDX', type: 'bigint', comment: 'serial' })
  idx?: number;

  @Column({ name: 'DEVICE_IDX', type: 'bigint', width: 20, comment: 'device index' })
  deviceIdx?: number;

  @Column({ name: 'DAYKEY', comment: 'day key' })
  dayKey?: number;
  @Column({ name: 'SENSING_DT', comment: 'sensing datetime' })
  sensingDt?: Date;
  @CreateDateColumn({ name: 'CREATE_DT' })
  createDt: Date;

  @BeforeInsert()
  private beforeInsert() {
    this.fill();

    const year = this.sensingDt.getFullYear();
    const month = String(this.sensingDt.getMonth() + 1).padStart(2, '0');
    const day = String(this.sensingDt.getDate()).padStart(2, '0');
    const ymd = `${year}${month}${day}`;
    this.dayKey = +ymd;

    this.createDt = new Date();
  }

  abstract fill();
  abstract validate(); // TODO: insert 여부 등 다시 처리
}
