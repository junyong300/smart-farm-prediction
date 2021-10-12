import { Entity, Column } from "typeorm";
import { SensingBaseEntity } from "./sensing-base.entity";

/**
 * 내부환경센서
 */
@Entity({ schema: 'env'})
export class Internal extends SensingBaseEntity {
    @Column()
    temp: number;
    @Column()
    humidity: number;
    @Column()
    wetBulbTemp: number;
    @Column()
    co2: number;
    @Column()
    dewPointTemp: number;
    @Column()
    hd: number;
    @Column()
    solarRadiation: number;
    /*
    @Column()
    rootTemp: number;
    @Column()
    rootHumidity: number;
    */
}