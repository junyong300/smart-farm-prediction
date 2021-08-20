import { Entity, Column } from "typeorm";
import { SensingBaseEntity } from "./sensing-base.entity";

@Entity({ schema: 'sensor'})
export class Internal extends SensingBaseEntity {
    @Column()
    temp: number;
}