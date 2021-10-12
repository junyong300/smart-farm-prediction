import { Entity, PrimaryColumn, BaseEntity } from "typeorm";

@Entity({ schema: 'env'})
export class SensingBaseEntity extends BaseEntity {
    @PrimaryColumn()
    deviceId!: number;
    @PrimaryColumn({ type: 'timestamptz' })
    sensingTime!: Date;
}