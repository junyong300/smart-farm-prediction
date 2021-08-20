import { Entity, PrimaryColumn, Column, BaseEntity } from "typeorm";

@Entity({ schema: 'sensor'})
export class SensingBaseEntity extends BaseEntity {
    @PrimaryColumn()
    idx!: string;
    @PrimaryColumn({ type: 'timestamptz' })
    sensing_dt!: Date;
    @Column({ type: 'timestamptz' })
    create_dt: Date;
}